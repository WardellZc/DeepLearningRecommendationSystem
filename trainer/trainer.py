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

    # 训练集训练，模型参数为2个
    def train_loop(self, train_user, train_item, train_rating):
        self.model.train()
        self.optimizer.zero_grad()
        self.predictions_train = self.model(train_user, train_item)
        self.train_loss = self.loss_fn(self.predictions_train, train_rating)
        self.train_loss.backward()
        self.optimizer.step()
        self.train_rating = train_rating

    # 验证集、测试集训练，模型参数为2个
    def valid_loop(self, valid_user, valid_item, valid_rating):
        self.model.eval()
        with torch.no_grad():
            self.predictions_valid = self.model(valid_user, valid_item)
            self.valid_loss = self.loss_fn(self.predictions_valid, valid_rating)
        self.valid_rating = valid_rating

    def test_loop(self, test_user, test_item, test_rating):
        self.model.eval()
        with torch.no_grad():
            self.predictions_test = self.model(test_user, test_item)
            self.test_loss = self.loss_fn(self.predictions_test, test_rating)
        self.test_rating = test_rating

    # 训练集训练，模型参数为1个
    def train_loop2(self, train_data, train_rating):
        self.model.train()
        self.optimizer.zero_grad()
        self.predictions_train = self.model(train_data)
        self.train_loss = self.loss_fn(self.predictions_train, train_rating)
        self.train_loss.backward()
        self.optimizer.step()
        self.train_rating = train_rating

    # 验证集、测试集训练，模型参数为1个
    def valid_loop2(self, valid_data, valid_rating):
        self.model.eval()
        with torch.no_grad():
            self.predictions_valid = self.model(valid_data)
            self.valid_loss = self.loss_fn(self.predictions_valid, valid_rating)
        self.valid_rating = valid_rating

    def test_loop2(self, test_data, test_rating):
        self.model.eval()
        with torch.no_grad():
            self.predictions_test = self.model(test_data)
            self.test_loss = self.loss_fn(self.predictions_test, test_rating)
        self.test_rating = test_rating

    # 训练集训练，模型参数为1个，输入要多一个掩码
    def train_loop3(self, train_matrix, mask):
        self.model.train()
        self.optimizer.zero_grad()
        self.predictions_train = self.model(train_matrix)
        # 使用掩码，使计算损失时不计算未评分部分,两个矩阵未评分的部分值都转化为0
        self.predictions_train = self.predictions_train * mask
        train_rating = train_matrix * mask
        # 计算损失
        self.train_loss = self.loss_fn(self.predictions_train, train_rating)
        self.train_loss.backward()
        self.optimizer.step()
        self.train_rating = train_matrix

    # 验证集、测试集训练，模型参数为1个，输入要多一个掩码
    def valid_loop3(self, valid_matrix, mask):
        self.model.eval()
        with torch.no_grad():
            self.predictions_valid = self.model(valid_matrix)
            # 使用掩码，使计算损失时不计算未评分部分,两个矩阵未评分的部分值都转化为0
            self.predictions_valid = self.predictions_valid * mask
            valid_rating = valid_matrix * mask
            self.valid_loss = self.loss_fn(self.predictions_valid, valid_rating)
        self.valid_rating = valid_matrix

    def test_loop3(self, test_matrix, mask):
        self.model.eval()
        with torch.no_grad():
            self.predictions_test = self.model(test_matrix)
            # 使用掩码，使计算损失时不计算未评分部分,两个矩阵未评分的部分值都转化为0
            self.predictions_test = self.predictions_test * mask
            test_rating = test_matrix * mask
            self.test_loss = self.loss_fn(self.predictions_test, test_rating)
        self.test_rating = test_matrix

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

    # 模型评估，将矩阵转化为一维张量 autorec
    def model_eval2(self, epoch):
        evaluator = Evaluator()

        # 将真实的评分矩阵，去除未评分内容，即值为0.5的部分
        self.train_rating = self.train_rating[self.train_rating != 0.5]
        self.valid_rating = self.valid_rating[self.valid_rating != 0.5]
        self.test_rating = self.test_rating[self.test_rating != 0.5]

        # 将预测的评分矩阵，去除未评分内容，即值为0的部分
        self.predictions_train = self.predictions_train[self.predictions_train != 0.0]
        self.predictions_valid = self.predictions_valid[self.predictions_valid != 0.0]
        self.predictions_test = self.predictions_test[self.predictions_test != 0.0]

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
