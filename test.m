% 加载训练好的网络
load('-mat','net\net.mat');
% 载入测试集并测试
namelist_features = dir("test_data\features_snr*.mat");
namelist_labels = dir("test_data\labels_snr*.mat");
net_error_rate_v = zeros(1,9);
for i = 1:length(namelist_labels)
    test_data_features = load(namelist_features(i).name);
    test_data_features = cell2mat(struct2cell(test_data_features));
    test_data_labels = load(namelist_labels(i).name);
    test_data_labels = cell2mat(struct2cell(test_data_labels));
    YPred = predict(net,test_data_features);
    for j = 1:numel(YPred)
        if YPred(j) > 0.5
            YPred(j) = 1;
        else
            YPred(j) = 0;
        end
    end
    YTest = test_data_labels;
    error_rate = sum(YPred ~= YTest, "all")/numel(YTest); 
    net_error_rate_v(i) = error_rate;
    fprintf('snr: %d  error rate = %.7f\n',i-1,error_rate);
end

disp(net_error_rate_v);
error_rate_v = cell2mat(struct2cell(load('-mat','test_data\error_rate_v.mat')));
error_rate_v1 = cell2mat(struct2cell(load('-mat','test_data\error_rate_v1.mat')));
draw_error_rate(error_rate_v, error_rate_v1, net_error_rate_v)

function draw_error_rate(error_rate_v, error_rate_v1, net_error_rate_v) %绘制各种信噪比下的误码率
    x = 0:8;
    xx = 0:0.1:8;
    y1 = spline(x,error_rate_v,xx);%三次样条数据插值
    y2 = spline(x,error_rate_v1,xx);
    y3 = spline(x,net_error_rate_v,xx);
    hold on
    plot(x,error_rate_v,'ko','MarkerFaceColor','r');
    plot(x,error_rate_v1,'ko','MarkerFaceColor','g');
    plot(x,net_error_rate_v,'ko','MarkerFaceColor','b');
    plot(xx,y1,'r',xx,y2,'g',xx,y3,'b');
    legend('coherent demodulate','incoherent demodulate','mlp demodulate');
    xlabel('snr/dB');
    ylabel('error rate');
    title('Plot of the relationship between error rate and snr:');
    hold off
end
