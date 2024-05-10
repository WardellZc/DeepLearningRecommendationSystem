import torch.nn
import sys

sys.path.append('../')
from evaluator.evaluator import Evaluator


class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None
        self.predictions_train = None
        self.predictions_valid = None
        self.predictions_test = None
        self.train_rating = None
        self.valid_rating = None
        self.test_rating = None

    def train_loop(self, *args, train_rating):
        self.model.train()
        self.optimizer.zero_grad()

        # 根据传入参数的数量决定模型的调用方式
        if len(args) == 2:  # 假设是(train_user, train_item)的情况
            train_user, train_item = args
            self.predictions_train = self.model(train_user, train_item)
        elif len(args) == 1:  # 假设是(train_data,)的情况
            (train_data,) = args  # 注意这里的逗号，它是解包元组的关键
            self.predictions_train = self.model(train_data)
        else:
            raise ValueError("Invalid number of arguments provided to train_loop")

        self.train_loss = self.loss_fn(self.predictions_train, train_rating)
        self.train_loss.backward()
        self.optimizer.step()
        self.train_rating = train_rating

    def valid_loop(self, *args, valid_rating):
        self.model.eval()
        with torch.no_grad():
            # 根据传入参数的数量调用模型
            if len(args) == 2:  # 假设是(valid_user, valid_item)的情况
                valid_user, valid_item = args
                self.predictions_valid = self.model(valid_user, valid_item)
            elif len(args) == 1:  # 假设是(valid_data,)的情况
                (valid_data,) = args  # 使用元组解包获取单个参数
                self.predictions_valid = self.model(valid_data)
            else:
                raise ValueError("Invalid number of arguments provided to valid_loop")

            # 计算验证集上的损失
            self.valid_loss = self.loss_fn(self.predictions_valid, valid_rating)

        # 存储验证集评分，可能用于后续操作
        self.valid_rating = valid_rating

    def test_loop(self, *args, test_rating):
        self.model.eval()
        with torch.no_grad():
            # 根据传入参数的数量调用模型
            if len(args) == 2:  # 假设是(test_user, test_item)的情况
                test_user, test_item = args
                self.predictions_test = self.model(test_user, test_item)
            elif len(args) == 1:  # 假设是(test_data,)的情况
                (test_data,) = args  # 使用元组解包获取单个参数
                self.predictions_test = self.model(test_data)
            else:
                raise ValueError("Invalid number of arguments provided to test_loop")

            # 计算测试集上的损失
            self.test_loss = self.loss_fn(self.predictions_test, test_rating)

        # 存储测试集评分，可能用于后续操作
        self.test_rating = test_rating

    # 训练集训练，模型参数为1个，输入要多一个掩码
    def train_loop2(self, train_matrix, mask):
        self.model.train()
        self.optimizer.zero_grad()
        self.predictions_train = self.model(train_matrix)
        # 使用掩码，取出评过分的数据
        self.predictions_train = self.predictions_train[mask]
        train_rating = train_matrix[mask]
        # 计算损失
        self.train_loss = self.loss_fn(self.predictions_train, train_rating)
        self.train_loss.backward()
        self.optimizer.step()
        self.train_rating = train_rating

    # 验证集、测试集训练，模型参数为1个，输入要多一个掩码
    def valid_loop2(self, valid_matrix, mask):
        self.model.eval()
        with torch.no_grad():
            self.predictions_valid = self.model(valid_matrix)
            # 使用掩码，取出评过分的数据
            self.predictions_valid = self.predictions_valid[mask]
            valid_rating = valid_matrix[mask]
            self.valid_loss = self.loss_fn(self.predictions_valid, valid_rating)
        self.valid_rating = valid_rating

    def test_loop2(self, test_matrix, mask):
        self.model.eval()
        with torch.no_grad():
            self.predictions_test = self.model(test_matrix)
            # 使用掩码，取出评过分的数据
            self.predictions_test = self.predictions_test[mask]
            test_rating = test_matrix[mask]
            self.test_loss = self.loss_fn(self.predictions_test, test_rating)
        self.test_rating = test_rating

    # 模型评估
    def model_eval(self, epoch):
        evaluator = Evaluator()
        train_eval = evaluator.eval(self.train_rating, self.predictions_train)
        valid_eval = evaluator.eval(self.valid_rating, self.predictions_valid)
        test_eval = evaluator.eval(self.test_rating, self.predictions_test)
        print(f"""
        Epoch {epoch + 1}:
          - Training Loss: {self.train_loss.item()}
          - Valid Loss: {self.valid_loss.item()}
          - Test Loss: {self.test_loss.item()}

          - Training Accuracy: {train_eval[0]}
          - Valid Accuracy: {valid_eval[0]}
          - Test Accuracy: {test_eval[0]}

          - Training Precision: {train_eval[1]}
          - Valid Precision: {valid_eval[1]}
          - Test Precision: {test_eval[1]}

          - Training Recall: {train_eval[2]}
          - Valid Recall: {valid_eval[2]}
          - Test Recall: {test_eval[2]}

          - Training F1 Score: {train_eval[3]}
          - Valid F1 Score: {valid_eval[3]}
          - Test F1 Score: {test_eval[3]}

          - Training ROC AUC Score: {train_eval[4]}
          - Valid ROC AUC Score: {valid_eval[4]}
          - Test ROC AUC Score: {test_eval[4]}
        """)
        # print(f"""
        # Epoch {epoch + 1}:
        #   - Training Loss: {self.train_loss.item()}- Training Accuracy: {train_eval[0]}- Training Precision: {train_eval[1]}- Training Recall: {train_eval[2]}- Training F1 Score: {train_eval[3]}- Training ROC AUC Score: {train_eval[4]}
        #   - Valid Loss: {self.valid_loss.item()}- Valid Accuracy: {valid_eval[0]}- Valid Precision: {valid_eval[1]} - Valid Recall: {valid_eval[2]}- Valid F1 Score: {valid_eval[3]}- Valid ROC AUC Score: {valid_eval[4]}
        #   - Test Loss: {self.test_loss.item()}- Test Accuracy: {test_eval[0]}- Test Precision: {test_eval[1]}- Test Recall: {test_eval[2]}- Test F1 Score: {test_eval[3]}- Test ROC AUC Score: {test_eval[4]}
        # """)
