from torch.utils.data import DataLoader
from transformers import FSMTTokenizer
from transformer import Transformer
from scheduler import LinearWarmupLR
from utils import WMTDataset, WMTTokenizedDataset, WMTSampler, make_collate
import argparse
import torch
from tqdm import tqdm
import wandb

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", default="allenai/wmt16-en-de-dist-6-1", type=str)
    parser.add_argument("--train_csv", default="./archive/wmt14_translate_de-en_train.csv", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--d_k", default=64, type=int)
    parser.add_argument("--d_v", default=64, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--hidden_dim", default=2048, type=int)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.98, type=float)
    parser.add_argument("--eps", default=1e-9, type=float)
    parser.add_argument("--warmup_steps", default=4000, type=float)
    parser.add_argument("--train_pkl", default="./tokenized_data.pkl", type=str)
    parser.add_argument("--train_meta_pkl", default="./tokenized_meta.pkl", type=str)
    parser.add_argument("--max_tokens", default=25000, type=int)
    parser.add_argument("--num_steps", default=100000, type=int)
    parser.add_argument("--val_pkl", default="", type=str)
    parser.add_argument("--val_meta_pkl", default="", type=str)
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--wandb_mode", default="disabled",
                        choices=["online", "offline", "disabled"], type=str)

    return parser

def cycle(loader):
    """Infinite iterator over the dataloader so we can train by step count."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, criterion, val_loader, device):
    """Average per-token loss over the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in val_loader:
        src, tgt = src.to(device), tgt.to(device)
        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        logits = model(src, decoder_input)
        # sum-reduce so batches of different token counts are weighted fairly.
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        num_tokens = (labels != criterion.ignore_index).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def train_one_step(step, model, optimizer, scheduler, criterion, batch, device):
    model.train()
    src, tgt = batch
    src, tgt = src.to(device), tgt.to(device)

    # Teacher forcing: decoder sees target shifted right, predicts shifted left.
    decoder_input = tgt[:, :-1]
    labels = tgt[:, 1:]

    logits = model(src, decoder_input)
    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lr = scheduler.get_last_lr()[0]
    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

    return loss.item()


def main():

    parser = get_parser()
    args = parser.parse_args()

    wandb.init(project="transformer-wmt", config=vars(args), mode=args.wandb_mode)

    tokenizer = FSMTTokenizer.from_pretrained(args.tokenizer_name)
    train_data = WMTTokenizedDataset(args.train_pkl)
    sampler = WMTSampler(args.max_tokens, args.train_meta_pkl)
    train_loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )

    val_loader = None
    if args.val_pkl:
        val_data = WMTTokenizedDataset(args.val_pkl)
        val_sampler = WMTSampler(args.max_tokens, args.val_meta_pkl, shuffle=False)
        val_loader = DataLoader(
            val_data,
            batch_sampler=val_sampler,
            collate_fn=make_collate(tokenizer.pad_token_id),
        )

    model = Transformer(
        args.d_model,
        args.hidden_dim,
        args.d_k,
        args.d_v,
        args.num_heads,
        tokenizer.vocab_size,
        args.num_layers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), betas=(args.beta_1, args.beta_2), eps=args.eps)
    scheduler = LinearWarmupLR(optimizer, args.d_model, args.warmup_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    wandb.watch(model, log="gradients", log_freq=100)

    data_iter = cycle(train_loader)
    progress = tqdm(range(1, args.num_steps + 1))
    for step in progress:
        batch = next(data_iter)
        loss = train_one_step(step, model, optimizer, scheduler, criterion, batch, device)
        progress.set_postfix(loss=loss)

        if val_loader is not None and step % args.eval_interval == 0:
            val_loss = evaluate(model, criterion, val_loader, device)
            wandb.log({"val/loss": val_loss}, step=step)
            progress.set_postfix(loss=loss, val_loss=val_loss)

    wandb.finish()

if __name__ == "__main__":
    main()
