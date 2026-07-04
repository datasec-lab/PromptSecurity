import torch
import torch.nn.functional as F
import numpy as np
import gc
from .util import *
from .bleuloss import batch_log_bleulosscnn_ae
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contextlib

stop_words = set(stopwords.words('english'))


def sim_score(model, y_logits, ref_embedding):
    """Compute similarity score between generated embeddings and reference."""
    embeddings_weight = model.get_input_embeddings().weight
    y_embeds = embed_inputs(embeddings_weight, y_logits, device=y_logits.device)
    # Disable cache to save memory:
    outputs = model(inputs_embeds=y_embeds, use_cache=False)
    gen_embedding = outputs.logits.mean(dim=1)
    return torch.nn.functional.cosine_similarity(gen_embedding, ref_embedding, dim=-1)


@torch.inference_mode()  # <--- reference embedding does not need gradients
def get_ref_embedding(model, x, device, tokenizer):
    """Get reference embedding for the input prompt (no gradient needed)."""
    x_ = tokenizer.encode(x)[1:]
    x_t = torch.tensor(x_, device=device, dtype=torch.long).unsqueeze(0)
    # Disable cache for large models
    outputs = model(x_t, use_cache=False)
    return outputs.logits.mean(dim=1)


def decode_paraphrase(attack, x, z, constraints=None):
    """
    Generate adversarial prompt using the COLD attack's paraphrase mode,
    with memory-optimization strategies to avoid GPU OOM.

    Also calculates the query number (an approximate count of model forward calls).

    Returns:
        tuple: (query_num, ppl_last, text, text_post, decoded_text, prompt_with_adv)
    """
    model = attack.target_model.model
    tokenizer = attack.tokenizer
    device = model.device
    args = attack  # use attack instance as args

    # Initialize query counter
    query_num = 0

    # Put model in eval mode and freeze params:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Example "bad words" list (unchanged):
    words = [
        'sorry', 'apologetic', 'regretful', 'unethical', 'immoral', 'dishonest', 'illegal',
        'unlawful', 'illicit', 'harmful', 'detrimental', 'injurious', 'afraid', 'fearful',
        'apprehensive', 'cannot', 'unable', 'apologize', 'regret', 'unable', 'incapable',
        'powerless', 'not', 'no', 'nay', 'responsible', 'accountable', 'liable', 'ethic',
        'morality', 'moral', 'legal', 'lawful', 'legitimate', 'ethical', 'moral', 'principled',
        'fulfill', 'accomplish', 'achieve', 'just', 'fair', 'equitable', 'trustworthy',
        'reliable', 'dependable', 'repentant', 'penitent', 'remorseful', 'unprincipled',
        'unscrupulous', 'corrupt', 'unlawful', 'prohibited', 'forbidden', 'noxious', 'destructive',
        'damaging', 'timid', 'terrified', 'panicked', 'unable', 'incompetent', 'incapacitated',
        'prohibited', 'banned', 'forbidden', 'deny', 'refuse', 'decline', 'answerable',
        'culpable', 'moral code', 'values', 'principles', 'legitimate', 'valid', 'authorized',
        'virtuous', 'righteous', 'honorable', 'satisfy', 'complete', 'execute', 'fair-minded',
        'impartial', 'equitable', 'reliable', 'trustable', 'faithful', 'invalid', 'safe', 'not',
        "can't", "but", "against", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    lowercase_words = [word.upper() for word in words]
    bad_words = ' '.join(words + lowercase_words)

    # Encode prefix
    x_ = tokenizer.encode(x)[1:]
    prefix_t = torch.tensor(x_, device=device, dtype=torch.long)
    prefix_onehot = one_hot(prefix_t, dimension=model.vocab_size)
    prefix_t = prefix_t.unsqueeze(0).repeat(args.batch_size, 1)
    prefix_onehot = prefix_onehot.repeat(args.batch_size, 1, 1)

    # Encode target z
    z_ = tokenizer.encode(z)[1:]
    z_t = torch.tensor(z_, device=device, dtype=torch.long)
    z_onehot = one_hot(z_t, dimension=model.vocab_size)
    z_onehot = z_onehot.repeat(args.batch_size, 1, 1)
    z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)

    # For BLEU-based similarity, remove stopwords from x
    x_words = word_tokenize(x)
    x_nonstop_words = [w.lower() for w in x_words if w.lower() not in stop_words and w.isalnum()]
    x_nonstop_words = ' '.join(x_nonstop_words)
    x_seq_t = torch.tensor(tokenizer.encode(x_nonstop_words.strip())[1:], device=device, dtype=torch.long)
    x_seq_t = x_seq_t.unsqueeze(0).repeat(args.batch_size, 1)

    # Create x_mask for BLEU
    x_mask_arr = np.zeros([tokenizer.vocab_size])
    x_mask_arr[tokenizer.encode(x_nonstop_words.strip())[1:]] = 1.0
    x_mask = torch.tensor(x_mask_arr, device=device).unsqueeze(0).unsqueeze(0)
    x_mask = x_mask.repeat(args.batch_size, args.length, 1)

    if args.verbose:
        print(f"x:\t|{tokenizer.decode(x_)}|\nz:\t|{tokenizer.decode(z_)}|\nlength:\t{args.length}")

    # Get reference embedding (no gradients)
    ref_embedding = get_ref_embedding(model, x, device, tokenizer).detach().to(torch.float16)
    query_num += 1  # for get_ref_embedding call

    # Compute prefix past (if applicable) without tracking gradients:
    if prefix_t.shape[1] > 1:
        with torch.inference_mode():
            prefix_outputs = model(prefix_t[:, :-1], use_cache=False)
            query_num += 1  # prefix past query
            prefix_model_past = prefix_outputs.past_key_values
    else:
        prefix_model_past = None

    # Initialize y_logits
    if args.init_mode == 'original':
        init_logits = initialize(model, prefix_t, args.length, args.init_temp, args.batch_size, device, tokenizer)
        query_num += 1  # count the initialization generation query
    else:
        init_logits = z_onehot / 0.01
        if args.length > init_logits.shape[1]:
            extra_rand = 10 * torch.rand([args.batch_size, args.length - init_logits.shape[1],
                                          tokenizer.vocab_size], device=device)
            init_logits = torch.cat([init_logits, extra_rand], dim=1)
        else:
            init_logits = init_logits[:, :args.length, :]

    if args.verbose:
        text, _, _ = get_text_from_logits(init_logits, tokenizer)
        for bi in range(args.batch_size):
            print(f"[initial]: {text[bi]}")

    # We only want gradient wrt epsilon
    y_logits = init_logits
    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits), requires_grad=True)
    optim = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.stepsize_iters, gamma=args.stepsize_ratio)

    # Main optimization loop
    for iter in range(args.num_iters):
        optim.zero_grad(set_to_none=True)

        # Main forward & loss computation in autocast if FP16 is desired
        with torch.autocast(device_type="cuda", dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
            y_logits_ = y_logits + epsilon
            soft_forward_y = y_logits_ / 0.001

            # soft_forward: one query call
            y_logits_t = soft_forward(
                model,
                prefix_onehot[:, -1:, :],
                soft_forward_y,
                args.topk,
                extra_mask=x_mask,
                x_past=prefix_model_past,
                bad_mask=None
            )
            query_num += 1

            # Optionally filter top_k
            if args.topk != 0:
                _, indices_t = torch.topk(y_logits_t, args.topk)
                mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
            else:
                mask_t = None

            # Fluency loss
            flu_loss = soft_nll(
                top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=x_mask),
                y_logits_ / args.input_lgt_temp
            )

            # Forward for cross-entropy w.r.t. z (one query call)
            xyz_logits, xy_length = soft_forward_xyz(
                model,
                prefix_onehot[:, -1:, :],
                y_logits_.detach(),
                z_onehot
            )
            query_num += 1

            bz = args.batch_size
            lg = xyz_logits.shape[1]
            st, ed = xy_length - 1, lg - 1
            xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
            z_logits = torch.cat([xyz_logits[bi * lg + st: bi * lg + ed, :] for bi in range(bz)], dim=0)

            c_loss_1 = F.cross_entropy(z_logits, z_t.view(-1), reduction='none')
            c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(dim=1)

            # BLEU-based paraphrase penalty
            c_loss_2 = batch_log_bleulosscnn_ae(
                decoder_outputs=top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=x_mask).transpose(0, 1),
                target_idx=x_seq_t,
                ngram_list=list(range(1, args.counterfactual_max_ngram + 1))
            )

            # Similarity with reference embedding
            c_loss_3 = sim_score(
                model,
                top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=x_mask),
                ref_embedding
            )

            loss = args.goal_weight * c_loss_1 + args.rej_weight * c_loss_2 + args.lr_nll_portion * flu_loss - c_loss_3
            loss = loss.mean()

        if iter < args.num_iters - 1:
            loss.backward()
            optim.step()
            scheduler.step()

        # (Optional) Logging every N steps (assume one additional query here)
        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or (iter + 1) == args.num_iters):
            with torch.no_grad():
                text, _, last_text_ids = decode_with_model_topk(
                    model,
                    y_logits.detach() + epsilon.detach(),
                    args.topk,
                    prefix_onehot[:, -1:, :],
                    prefix_model_past,
                    tokenizer,
                    extra_mask=x_mask
                )
            query_num += 1
            for bi in range(args.batch_size):
                print(f"[Iter {iter + 1}/{args.num_iters}] loss={loss.item():.4f}, "
                      f"c_loss_1={c_loss_1[bi]:.4f}, c_loss_2={c_loss_2[bi]:.4f}, "
                      f"c_loss_3={c_loss_3[bi]:.4f}, |{text[bi]}|")

        # Optionally inject noise or detach
        if iter < args.num_iters - 1:
            large_noise_iters = [int(x) for x in args.large_noise_iters.split(',')]
            large_gs_stds = [float(x) for x in args.large_gs_std.split(',')]
            noise_std = 0.0
            if (iter % args.noise_iters) == 0:
                noise_last = True
                for ni, it_val in enumerate(large_noise_iters):
                    if iter < it_val:
                        noise_last = False
                        break
                noise_std = args.gs_std if noise_last else large_gs_stds[min(ni, len(large_gs_stds) - 1)]
                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(), device=device,
                                     requires_grad=False)
                y_logits = (y_logits + noise).detach()
            else:
                y_logits = y_logits.detach()

        del y_logits_t, xyz_logits, z_logits, flu_loss
        del c_loss_1, c_loss_2, c_loss_3, loss
        torch.cuda.empty_cache()
        gc.collect()

    # Final decoding after optimization
    with torch.no_grad():
        final_logits = y_logits.detach() + epsilon.detach()
        text, _, last_text_ids = decode_with_model_topk(
            model,
            final_logits,
            args.topk,
            prefix_onehot[:, -1:, :],
            prefix_model_past,
            tokenizer,
            extra_mask=x_mask,
        )
        query_num += 1

    # Final call to model.generate (count one query per batch element)
    decoded_text = []
    for bi in range(args.batch_size):
        prompt_final = x + " " + text[bi]
        input_ids = tokenizer.encode(prompt_final, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                temperature=0.7,
                max_length=512,
                do_sample=True,
                top_k=args.topk,
                use_cache=False
            )
        query_num += 1
        output_ids = output_ids[:, input_ids.shape[1]:]
        text_dec = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decoded_text.append(text_dec.strip())

    # Compute perplexity over the final tokens (one query)
    with torch.no_grad():
        last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids, use_cache=False).loss
    query_num += 1
    ppl_last = float(np.exp(last_rank_loss.detach().cpu().numpy()))

    prompt_with_adv = [x + " " + t for t in text]

    # Return query number as the first output
    return query_num, ppl_last, text, text, decoded_text, prompt_with_adv

