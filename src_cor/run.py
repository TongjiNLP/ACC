# 运行主程序

# 外部库
import argparse

# 文件库
from argpaser import getargs
from utils import *
from learning import train,evaluate,test

if(not os.path.exists('./logs')):
    os.mkdir('./logs')

def main():
    # 获取参数
    parser = argparse.ArgumentParser()
    args=getargs(parser)
    # 准备model
    args,data_processor,tokenizer,model=initialize(args)
    # 训练
    if args.do_train:
        train(args,data_processor,tokenizer,model)
    # 预测
    if args.do_eval:
        results=evaluate(args,data_processor,tokenizer)
        with open(os.path.join(args.output_dir,"results.json"),"w",encoding="utf-8",newline="") as f:
            json.dump(results,f,indent=4,ensure_ascii=False)
        K=min(len(results),3)
        best_checkpoints=sorted(results,key=lambda x:x[1]['exact'],reverse=True)[:K]
        print(best_checkpoints)
        # for each_dir in os.listdir(args.predict_dir):
        #     if not each_dir in best_checkpoints:
        #         os.removedirs(os.path.join(args.predict_dir,each_dir))
    # # 测试
    # if args.do_eval and args.do_test:
    #     test(args,data_processor,tokenizer,checkpoints)

if __name__=="__main__":
    main()