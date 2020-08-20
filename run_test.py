# -*- coding: utf-8 -*-
# @Time   : 2020/7/20 20:32
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test.py


from trainer import Trainer
from utils import init_logger, get_model
from config import Config
from data import Dataset, data_preparation


def whole_process(config_file='properties/overall.config', config_dict=None):
    """
    初始化 config
    """
    config = Config(config_file, config_dict)
    config.init()
    """
    
    初始化 logger
    """
    init_logger(config)
    """
    初始化 dataset
    """
    dataset = Dataset(config)
    print(dataset)


    """
    初始化 model
    """
    model = get_model(config)(config, dataset).to(config['device'])
    print(model)

    """
    生成 训练/验证/测试 数据
    """

    train_data, test_data, valid_data = data_preparation(config, model, dataset)
    config.init()
    """
    初始化 trainer
    """
    trainer = Trainer(config, model)

    """
    训练
    """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """
    测试
    """
    test_result = trainer.evaluate(test_data)

    print('best valid result:', best_valid_result)
    print('test result: ', test_result)
    #model.dump_paramaters()


def run_test():
    whole_process()


if __name__ == '__main__':
    run_test()
