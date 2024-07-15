import torch
import LSTMmodel as lstm
import load_data


def predict(model, input_data, scaler):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器来关闭梯度计算，以便在推断过程中不计算梯度。
        input_data = scaler.transform([input_data])
        # 将输入数据转换为torch张量（tensor），并进行一些维度调整，最后将其移动到GPU上
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to('cuda')
        output = model(input_tensor)  # 将输入数据传递给模型，获取输出结果
        return output.item()


def main():
    # 加载数据
    _, _, custom_dataset = load_data.load_data()

    # 构建模型，参数需要与训练的时候相同
    input_size = len(custom_dataset.X.columns)
    hidden_size = 32
    output_size = 1
    num_layers = 4
    model = lstm.LSTMModel(input_size, hidden_size, output_size, num_layers).to('cuda')

    # 加载模型
    model.load_state_dict(torch.load('D:/WORK/T3low/影评项目预测/LSTM模型/训练好的模型/lstm_model_580.pth'))

    # 输入预测数据
    # 上座率(%)	场均人次	     票房占比    已上映天数
    # 排片场次	排片占比(%)   当日总出票   当日总场次
    # 出品国家	电影类别1	    电影类别2	    电影类别3
    # 电影评分	男性占比(%)	女性占比(%)	节假日
    input_data = [0.123, 14.4,   0.142, 2,
                  75130, 0.158, 746.6,  47.3,
                  3,     1,     5,    8,
                  9.1,   0.35, 0.65, 1]  # 示例输入数据

    # 进行预测
    prediction = predict(model, input_data, custom_dataset.scaler)

    print(f'模型预测今日票房结果: {prediction:.1f}万')


if __name__ == "__main__":
    main()
