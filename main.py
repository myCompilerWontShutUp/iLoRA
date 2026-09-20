import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from model.model_interface import MInterface
from data.data_interface import DInterface
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
from SASRecModules_ori import *
from transformers import LlamaForCausalLM, LlamaTokenizer

def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='metric',
        dirpath=args.ckpt_dir,
        filename='{epoch:02d}-{metric:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        #train_time_interval=args.val_check_interval
        every_n_epochs=1
    ))
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks

def main(args):
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')

        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints from {}".format(args.ckpt_path))

    data_module = DInterface(llm_tokenizer=model.llama_tokenizer,**vars(args))

    args.max_steps=len(data_module.trainset) * args.max_epochs // (args.accumulate_grad_batches * args.batch_size)
    logger = TensorBoardLogger(save_dir='./log/', name=args.log_dir)
    args.callbacks = load_callbacks(args)
    args.logger = logger
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    trainer = Trainer.from_argparse_args(args)

    if args.auto_lr_find:
        lr_finder=trainer.tuner.lr_find(model=model, datamodule=data_module, min_lr=1e-10, max_lr=1e-3, num_training=100)
        fig=lr_finder.plot(suggest=True)
        fig_path="lr_finder.png"
        fig.savefig(fig_path)
        print("Saving to {}".format(fig_path))
        model.hparams.lr=lr_finder.suggestion()

    if args.mode == 'train':
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()

    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices', default=-1, type=list)
    parser.add_argument('--precision', default=16, type=int)
    parser.add_argument('--amp_backend', default="native", type=str)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--accumulate_grad_batches', default=32, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)

    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-6, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-6, type=float)

    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    parser.add_argument('--dataset', default='steam_data', type=str)
    parser.add_argument('--data_dir', default='LLaRA_MOE/data/ref/steam', type=str)
    parser.add_argument('--model_name', default='mlp_projector', type=str)
    parser.add_argument('--loss', default='lm', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--ckpt_dir', default='LLaRA_MOE/checkpoints/steam/', type=str)
    parser.add_argument('--log_dir', default='steam_logs', type=str)
    
    parser.add_argument('--rec_size', default=64, type=int)
    parser.add_argument('--padding_item_id', default=3581, type=int)
    parser.add_argument('--llm_path', default='/home/hexngroup/kongxy/llama2_7b_hf', type=str)
    parser.add_argument('--rec_model_path', default='LLaRA_MOE/rec_model/SASRec_steam.pt', type=str)
    parser.add_argument('--prompt_path', default='LLaRA_MOE/prompt/game.txt', type=str)
    parser.add_argument('--output_dir', default='LLaRA_MOE/output/steam_moe/', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--rec_embed', default="SASRec", choices=['SASRec', 'Caser','GRU'], type=str)

    parser.add_argument('--aug_prob', default=0.5, type=float)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--metric', default='hr', choices=['hr'], type=str)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--save', default='part', choices=['part', 'all'], type=str)
    parser.add_argument('--cans_num', default=20, type=int)

    # Finetuning
    parser.add_argument('--llm_tuning', default='moelora', choices=['lora', 'freeze','freeze_lora', 'moelora'], type=str)
    parser.add_argument('--peft_dir', default=None, type=str)
    parser.add_argument('--peft_config', default=None, type=str)
    parser.add_argument('--lora_r', default=8, type=float)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--num_moe', default=4, type=int)
    parser.add_argument('--gating', default='Dense', type=str)
    
    parser.add_argument('--local_rank', default=3, type=int)
    
    parser.add_argument('--if_rand', default=False, type=bool)
    
    parser.add_argument('--router', default='unshare', choices=['share', 'unshare'], type=str)

    # exp/gate-cluster: routing-mode experiment additions. Defaults reproduce the original
    # dynamic-routing baseline exactly; see experiments/IMPLEMENTATION_NOTES.md section 6/11.
    parser.add_argument('--routing_mode', default='dynamic', choices=['dynamic', 'cluster_hard'], type=str)
    parser.add_argument('--cluster_model_path', default=None, type=str)
    parser.add_argument('--cluster_assignment_dir', default='results/clustering', type=str)
    parser.add_argument('--cluster_expert_mapping', default='0:0,1:1,2:2,3:3', type=str)
    parser.add_argument('--export_gates', default=False, action='store_true')
    parser.add_argument('--results_dir', default='results/baseline', type=str)

    args = parser.parse_args()
    
    if 'movielens' in args.data_dir:
        args.padding_item_id = 1682
    elif 'steam' in args.data_dir:
        args.padding_item_id = 3581
    elif 'lastfm' in args.data_dir:
        args.padding_item_id = 4606

    main(args)
