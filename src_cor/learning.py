# 训练、验证、测试用
from utils import *
from dataloader import load_dataset
import squad_evaluation
import argparse

# acc
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# 返回单个checkpoint的ACC，loss
def eval_checkpoint(args,model,tokenizer,dataset,examples,features,prefix,run_evaluate=True):
    # 采样器准备
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ## 多GPU
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(dataset))
    print("  Batch size = %d", args.eval_batch_size)

    # 存放结果
    all_results=[]

    # 开始测试
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:
        # 准备模型输入
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        token_type_ids=None
        if args.model_type in ['bert', 'xlnet', 'albert']:
            token_type_ids=batch[2]
        # 计算
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': token_type_ids
            }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)
            # 处理结果
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                if args.model_type in ['xlnet', 'xlm']:
                    # XLNet uses a more complex post-processing procedure
                    result = RawResultExtended(unique_id            = unique_id,
                                            start_top_log_probs  = to_list(outputs[0][i]),
                                            start_top_index      = to_list(outputs[1][i]),
                                            end_top_log_probs    = to_list(outputs[2][i]),
                                            end_top_index        = to_list(outputs[3][i]),
                                            cls_logits           = to_list(outputs[4][i]))
                else:
                    result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)        
    # 后处理并计算分数
    # Compute predictions
    os.mkdir(args.predict_dir) if not os.path.exists(args.predict_dir) else None
    output_dir=os.path.join(args.predict_dir,prefix)
    os.mkdir(output_dir) if not os.path.exists(output_dir) else None
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
    # Evaluate with the official SQuAD script
    if run_evaluate:
        squad_evaluation.OPTS=argparse.Namespace(
            data_file=os.path.join(args.data_dir,"dev.json"),
            pred_file=output_prediction_file,
            out_file=None,
            na_prob_file=output_null_log_odds_file,
            na_prob_thresh=1.0,
            out_image_dir=None
        )
        results=squad_evaluation.main()
        return results
    return None

def train(args,processor,tokenizer,model):
    # 读取数据
    dataset=load_dataset(args,processor,tokenizer,data_type="train",output_examples=False)
    eval_dataset,eval_examples,eval_features=load_dataset(args,processor,tokenizer,data_type="dev",output_examples=True)
    # 相关内容准备
    ## summary writer准备
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    ## dataloader 准备
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    print("train_batch_size={}".format(args.train_batch_size))

    train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    ##     
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    ## 多卡调整和分布式计算调整
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # 输出训练信息，准备相关数据
    print("***** Running training *****")
    print("  Num examples = %d", len(dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_exact=-1.0
    # best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    # 正式训练
    ## 第一层循环：每个epoch
    for _ in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator=train_dataloader
        ## 第二层循环：按batch取数据
        for step, batch in enumerate(epoch_iterator):
            model.train()
            ### 取出batch，调整输入
            batch = tuple(t.to(args.device) for t in batch)
            token_type_ids=None
            if args.model_type in ['bert', 'xlnet', 'albert']:
                token_type_ids=batch[2]
            
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  None if args.model_type == 'xlm' else batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
            ### 送入模型，利用forward取出loss，并进行调整
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            ### 如果是有多步累积loss，则先除一下
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            ### 指定步数后反向传播
            if (step + 1) % args.gradient_accumulation_steps == 0:
            
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(f"Average loss: {str((tr_loss - logging_loss)/args.logging_steps)} at global step: {str(global_step)}")
                logging_loss = tr_loss
            #### 保存模型
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                result = eval_checkpoint(args, model,tokenizer, eval_dataset, eval_examples,eval_features,prefix="best",run_evaluate=True)
                if result["exact"]>best_exact:
                    best_exact=result["exact"]
                    best_steps=global_step
                    # Save model checkpoint
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, WEIGHTS_NAME))
                    print(f"Saving model checkpoint to {output_dir}")



            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    print(f"best exact: {best_exact} , best steps: {best_steps}")

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

# 返回top-k的checkpoint
def evaluate(args,processor,tokenizer):
    # 取出数据集和模型
    dataset,examples,features=load_dataset(args,processor,tokenizer,data_type="dev",output_examples=True)
    _,model_class,_ = MODEL_CLASSES[args.model_type]
    # 找出checkpoints
    # checkpoints = [args.output_dir]
    # if args.eval_all_checkpoints:
    #     checkpoints = list(
    #         os.path.dirname(c) for c in sorted(
    #             glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True), key=lambda x: int(
    #             re.findall(".*checkpoint-(.*)/.*", x)[0] if len(re.findall(".*checkpoint-(.*)/.*", x)) > 0 else 0)
    #             )
    #         )
    checkpoints=[os.path.join(args.output_dir,"checkpoint-best")]
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    print("Evaluate the following checkpoints: %s", checkpoints)
    # 一一验证
    results = []
    for checkpoint in checkpoints:
        # global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = eval_checkpoint(args, model,tokenizer, dataset, examples,features,prefix=prefix,run_evaluate=True)
        result=dict(result)
        # result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.append([checkpoint,result])
    return results

# 对指定的checkpoints进行测试,获取答案
def test(args,processor,tokenizer,checkpoints):
    # 取出数据集和模型
    dataset,examples,features=load_dataset(args,processor,tokenizer,data_type="test",output_examples=True)
    _,model_class,_ = MODEL_CLASSES[args.model_type]
    for checkpoint in checkpoints:
        prefix = checkpoint.split('/')[-1]+"-test" if checkpoint.find('checkpoint') != -1 else "test"
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        _ = eval_checkpoint(args, model, tokenizer,dataset, examples,features,prefix=prefix,run_evaluate=False)
