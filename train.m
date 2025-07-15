% 数据处理
train_data_features = load("train_data\features.mat");
train_data_features = cell2mat(struct2cell(train_data_features));
train_data_labels = load("train_data\labels.mat");
train_data_labels = cell2mat(struct2cell(train_data_labels));
rowrank = randperm(size(train_data_features, 1));
train_data_features_shuffle = train_data_features(rowrank,:);
train_data_labels_shuffle = train_data_labels(rowrank,:);
% disp(length(find((train_data_features_shuffle-train_data_features)~=0)));
% disp(length(find((train_data_labels_shuffle-train_data_labels)~=0)));

val_data_features = load("val_data\features.mat");
val_data_features = cell2mat(struct2cell(val_data_features));
val_data_labels = load("val_data\labels.mat");
val_data_labels = cell2mat(struct2cell(val_data_labels));

% 网络结构
layers = [
    featureInputLayer(1024)
    fullyConnectedLayer(512)
    reluLayer
    
    fullyConnectedLayer(256)
    reluLayer
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(16)
    sigmoidLayer
    regressionLayer];

% 设置训练参数
options = trainingOptions('adam', ...    %优化器
    'InitialLearnRate',0.01, ...         %初始学习率
    'LearnRateSchedule','piecewise',...  %训练学习率
    'LearnRateDropFactor', 0.9, ...      %下降因子
    'LearnRateDropPeriod', 1, ...        %每一轮学习率改变
    'MaxEpochs',20, ...                  %最大学习整个数据集的次数
    'MiniBatchSize',128,...              %每次学习样本数
    'Shuffle','every-epoch', ...         %每一轮随机打乱顺序
    'ValidationData', {val_data_features, val_data_labels}, ...%验证集
    'ValidationFrequency',10, ...        %验证频率，几个batchsize后验证一次
    'Verbose',true, ...                  %在命令行窗口显示实时训练进程
    'Plots','training-progress' ...      %画出整个训练过程
    );

% 训练神经网络并打印运行时长
tic
net = trainNetwork(train_data_features_shuffle,train_data_labels_shuffle,layers,options);
toc

% 保存网络
save 'net\net.mat' net


