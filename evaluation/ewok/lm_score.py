import torch
import torch.nn.functional as F


@torch.no_grad()
def rank_mlm(sentences, model, tokenizer, device, batch_size, temperatures=None):
    mask_index = tokenizer.token_to_id("[MASK]")
    cls_index = torch.tensor([tokenizer.token_to_id("[CLS]")])
    sep_index = torch.tensor([tokenizer.token_to_id("[SEP]")])
    pad_index = tokenizer.token_to_id("[PAD]")

    context_sentences = sentences[0]
    target_sentences = sentences[2:]

    sentences = [" ".join([context_sentences, ts]) for ts in target_sentences]

    sentences = [torch.tensor(tokenizer.encode(s, add_special_tokens=False).ids) for s in sentences]
    context_sentences = torch.tensor(tokenizer.encode(context_sentences, add_special_tokens=False).ids)
    context_length = 0

    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    labels = torch.cat([s[context_length:] for i, s in enumerate(sentences)]).unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device)

    def prepare(tokens, context_length, padding: int):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - context_length - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[context_length+1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = True
        return input_ids, attention_mask

    max_length = max(s.size(0)for s in sentences)
    input_ids, attention_masks = zip(*[prepare(s, context_length, max_length - s.size(0)) for i, s in enumerate(sentences)])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)

    indices = [torch.arange(context_length + 1, 1 + len(s), device=device) for i, s in enumerate(sentences)]
    indices = torch.cat(indices, dim=0)

    total_score = []

    for b in range(input_ids.size(0) // batch_size + 1):
        logits = model(
            input_ids[b * batch_size : (b+1) * batch_size, :].t().contiguous(),
            attention_mask[b * batch_size : (b+1) * batch_size, :].contiguous().unsqueeze(1),
        ).transpose(0, 1)

        logits = torch.gather(
            logits,
            dim=1,
            index=indices[b * batch_size : (b+1) * batch_size].reshape(-1, 1, 1).expand(-1, -1, logits.size(-1))
        ).squeeze(1)
        logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels[:, b * batch_size : (b+1) * batch_size, :], dim=-1).squeeze(-1)
        total_score.append(log_p)

    total_score = torch.cat(total_score, dim=1)

    log_ps, offset = [], 0
    for i in range(len(sentences)):
        from_index = offset
        to_index = offset + (sentences[i].size(0) - context_length)
        log_ps.append(total_score[:, from_index:to_index].sum(-1))
        offset = to_index

    ranking = torch.argsort(torch.stack(log_ps, dim=1), dim=1, descending=True).tolist()
    return ranking[int(1.0 / 0.05)], ranking


@torch.no_grad()
def rank_causal(sentences, model, tokenizer, device, batch_size, temperatures=None):
    cls_index = torch.tensor([tokenizer.token_to_id("<s>")])
    pad_index = tokenizer.token_to_id("<pad>")

    context_sentences = sentences[0]
    target_sentences = sentences[2:]

    sentences = [" ".join([context_sentences, ts]) for ts in target_sentences]

    sentences = [torch.tensor(tokenizer.encode(s, add_special_tokens=False).ids) for s in sentences]
    context_sentences = torch.tensor(tokenizer.encode(context_sentences, add_special_tokens=False).ids)
    context_length = 0

    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    labels = [s[context_length:].unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device) for i, s in enumerate(sentences)]

    def prepare(tokens, padding: int):
        input_ids = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        attention_mask = torch.ones(input_ids.size(0), input_ids.size(0), dtype=torch.bool, device=device).triu(diagonal=1)
        return input_ids, attention_mask

    max_length = max(s.size(0) for s in sentences)
    input_ids, attention_masks = zip(*[prepare(s, max_length - s.size(0)) for s in sentences])

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    indices = [torch.arange(context_length, len(s), device=device) for i, s in enumerate(sentences)]

    logits = model(input_ids.t(), attention_masks).transpose(0, 1)

    def calc_log_p(logits, labels, temperatures):
        logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels, dim=-1).squeeze(-1)
        return log_p

    log_ps = []
    for i, logit in enumerate(logits):
        log_ps.append(calc_log_p(logit[indices[i]], labels[i], temperatures).sum(-1))

    ranking = torch.argsort(torch.stack(log_ps, dim=1), dim=1, descending=True).tolist()
    return ranking[int(1.0 / 0.05)], ranking


@torch.no_grad()
def rank_mlm_shift(sentences, model, tokenizer, device, batch_size, temperatures=None):
    mask_index = tokenizer.token_to_id("<mask>")
    cls_index = torch.tensor([tokenizer.token_to_id("<s>")])
    pad_index = tokenizer.token_to_id("<pad>")

    context_sentences = sentences[0]
    target_sentences = sentences[2:]

    sentences = [" ".join([context_sentences, ts]) for ts in target_sentences]

    sentences = [torch.tensor(tokenizer.encode(s, add_special_tokens=False).ids) for s in sentences]
    context_sentences = torch.tensor(tokenizer.encode(context_sentences, add_special_tokens=False).ids)
    context_length = 0

    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    labels = torch.cat([s[context_length:] for s in sentences]).unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device)

    def prepare(tokens, context_length, padding: int):
        tokens = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - context_length - 1 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[context_length + 1:(-padding if padding > 0 else None), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = True
        return input_ids, attention_mask

    max_length = max(s.size(0) for s in sentences)
    input_ids, attention_masks = zip(*[prepare(s, context_length, max_length - s.size(0)) for s in sentences])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)

    indices = [torch.arange(context_length, len(s), device=device) for s in sentences]
    indices = torch.cat(indices, dim=0)

    total_score = []

    for b in range(input_ids.size(0) // batch_size + 1):
        logits = model(
            input_ids[b * batch_size : (b+1) * batch_size, :].t().contiguous(),
            attention_mask[b * batch_size : (b+1) * batch_size, :].contiguous().unsqueeze(1),
        ).transpose(0, 1)

        logits = torch.gather(
            logits,
            dim=1,
            index=indices[b * batch_size : (b+1) * batch_size].reshape(-1, 1, 1).expand(-1, -1, logits.size(-1))
        ).squeeze(1)
        logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels[:, b * batch_size : (b+1) * batch_size, :], dim=-1).squeeze(-1)
        total_score.append(log_p)

    total_score = torch.cat(total_score, dim=1)

    log_ps, offset = [], 0
    for i in range(len(sentences)):
        from_index = offset
        to_index = offset + (sentences[i].size(0) - context_length)
        log_ps.append(total_score[:, from_index:to_index].sum(-1))
        offset = to_index

    ranking = torch.argsort(torch.stack(log_ps, dim=1), dim=1, descending=True).tolist()
    return ranking[int(1.0 / 0.05)], ranking


@torch.no_grad()
def rank_fused(sentences, model, tokenizer, device, batch_size, temperatures=None):
    mask_index = tokenizer.token_to_id("␥")
    cls_index = torch.tensor([tokenizer.token_to_id("␂")])
    pad_index = tokenizer.token_to_id("␢")

    context_sentences = sentences[0]
    target_sentences = sentences[2:]

    sentences = [" ".join([context_sentences, ts]) for ts in target_sentences]

    sentences = [torch.tensor(tokenizer.encode(s, add_special_tokens=False).ids) for s in sentences]
    context_sentences = torch.tensor(tokenizer.encode(context_sentences, add_special_tokens=False).ids)
    context_length = 0

    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    labels = [s[context_length:].unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device) for i, s in enumerate(sentences)]

    def prepare(tokens, padding: int):
        input_ids = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        attention_mask = torch.ones(input_ids.size(0), input_ids.size(0), dtype=torch.bool, device=device).triu(diagonal=1)
        return input_ids, attention_mask

    max_length = max(s.size(0) for s in sentences)
    input_ids, attention_masks = zip(*[prepare(s, max_length - s.size(0)) for s in sentences])

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    indices = [torch.arange(context_length, len(s), device=device) for s in sentences]

    logits = model(input_ids.t(), attention_masks).transpose(0, 1)

    logits_pos = logits[0][indices[0]]
    logits_pos = logits_pos.unsqueeze(0) / temperatures.view(-1, 1, 1)
    log_p_pos = F.log_softmax(logits_pos, dim=-1)
    log_p_pos = log_p_pos.gather(index=labels[0], dim=-1).squeeze(-1)

    logits_neg = logits[1][indices[1]]
    logits_neg = logits_neg.unsqueeze(0) / temperatures.view(-1, 1, 1)
    log_p_neg = F.log_softmax(logits_neg, dim=-1)
    log_p_neg = log_p_neg.gather(index=labels[1], dim=-1).squeeze(-1)

    total_score_causal = torch.cat([log_p_pos, log_p_neg], dim=1)

    labels = torch.cat([s[context_length:] for s in sentences]).unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device)

    def prepare_mlm(tokens, context_length, padding: int):
        tokens = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - context_length - 1 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[context_length + 1:(-padding if padding > 0 else None), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = True
        return input_ids, attention_mask

    max_length = max(s.size(0) for s in sentences)
    input_ids, attention_masks = zip(*[prepare_mlm(s, context_length, max_length - s.size(0)) for s in sentences])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)

    indices = [torch.arange(context_length, len(s), device=device) for s in sentences]
    indices = torch.cat(indices, dim=0)

    total_score_mlm = []

    for b in range(input_ids.size(0) // batch_size + 1):
        logits = model(
            input_ids[b * batch_size : (b+1) * batch_size, :].t().contiguous(),
            attention_mask[b * batch_size : (b+1) * batch_size, :].contiguous().unsqueeze(1),
        ).transpose(0, 1)

        logits = torch.gather(
            logits,
            dim=1,
            index=indices[b * batch_size : (b+1) * batch_size].reshape(-1, 1, 1).expand(-1, -1, logits.size(-1))
        ).squeeze(1)
        logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels[:, b * batch_size : (b+1) * batch_size, :], dim=-1).squeeze(-1)
        total_score_mlm.append(log_p)

    total_score_mlm = torch.cat(total_score_mlm, dim=1)

    total_score = total_score_causal + total_score_mlm

    log_ps, offset = [], 0
    for i in range(len(sentences)):
        from_index = offset
        to_index = offset + (sentences[i].size(0) - context_length)
        log_ps.append(total_score[:, from_index:to_index].sum(-1))
        offset = to_index

    ranking = torch.argsort(torch.stack(log_ps, dim=1), dim=1, descending=True).tolist()
    return ranking[int(1.0 / 0.05)], ranking


@torch.no_grad()
def rank_prefix(sentences, model, tokenizer, device, batch_size, temperatures=None):
    cls_index = torch.tensor([tokenizer.token_to_id("<s>")])
    pad_index = tokenizer.token_to_id("<pad>")

    context_sentences = sentences[0]
    target_sentences = sentences[2:]

    sentences = [" ".join([context_sentences, ts]) for ts in target_sentences]

    sentences = [torch.tensor(tokenizer.encode(s, add_special_tokens=False).ids) for s in sentences]
    context_sentences = torch.tensor(tokenizer.encode(context_sentences, add_special_tokens=False).ids)
    context_length = 0
    prefix_length = context_sentences.size(0)

    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    labels = [s[context_length:].unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device) for i, s in enumerate(sentences)]

    def prepare(tokens, prefix_length: int, padding: int):
        input_ids = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        attention_mask = torch.ones(input_ids.size(0), input_ids.size(0), dtype=torch.bool, device=device).triu(diagonal=1)
        attention_mask[:prefix_length+1, :prefix_length+1] = False
        return input_ids, attention_mask

    max_length = max(s.size(0) for s in sentences)
    input_ids, attention_masks = zip(*[prepare(s, prefix_length, max_length - s.size(0)) for s in sentences])

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    indices = [torch.arange(context_length, len(s), device=device) for i, s in enumerate(sentences)]

    logits = model(input_ids.t(), attention_masks).transpose(0, 1)

    def calc_log_p(logits, labels, temperatures):
        logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels, dim=-1).squeeze(-1)
        return log_p

    log_ps = []
    for i, logit in enumerate(logits):
        log_ps.append(calc_log_p(logit[indices[i]], labels[i], temperatures).sum(-1))

    ranking = torch.argsort(torch.stack(log_ps, dim=1), dim=1, descending=True).tolist()
    return ranking[int(1.0 / 0.05)], ranking
