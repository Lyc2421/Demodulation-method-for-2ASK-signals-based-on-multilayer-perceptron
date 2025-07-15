classdef ASK
    properties
        meta_num = 16        %一个样本中的码元数量
        meta_sampling = 64   %一个码元采样点数
        RB = 1               %码元速率
        fc = 5               %载波速率
        fs                   %采样频率
        sum_sampling         %一个样本中总的采样点数
        t                    %时间向量
    end
    
    methods
        function ask = ASK(meta_num, meta_sampling, RB, fc)
            ask.meta_num = meta_num;
            ask.meta_sampling = meta_sampling;
            ask.RB = RB;
            ask.fc = fc;
            ask.fs = RB*meta_sampling;
            ask.sum_sampling = meta_num*meta_sampling;
            ask.t = linspace(0, meta_num, ask.sum_sampling);
        end

        function m = metas(ask) %生成码元序列
            m = round(rand(1, ask.meta_num));
        end

        function s = signal(ask, m) %生成原始信号
            s = ask.t;
            for i = 1:ask.meta_num
                if m(i)<1
                    s(ask.meta_sampling*(i-1)+1:ask.meta_sampling*i) = 0;
                else
                    s(ask.meta_sampling*(i-1)+1:ask.meta_sampling*i) = 1;
                end
            end
        end

        function s_ask = ask_modulate(ask, s) %调制
            carry_wave = cos(2*pi*ask.fc*ask.t);
            s_ask = carry_wave.*s; 
        end
        
        function s_ask_n = ask_modulate_noise(ask, s, snr) %调制加噪声
            carry_wave = cos(2*pi*ask.fc*ask.t);
            s_ask = carry_wave.*s;
            s_ask_n = awgn(s_ask, snr);
        end

        function [s_ask_d, metas_d] = ask_coherent_demodulate(ask, s_ask_n) %相干解调
            s_ask_d = bandpass(s_ask_n, [ask.fc-ask.RB, ask.fc+ask.RB], ask.fs);
            s_ask_d = s_ask_d.*cos(2*pi*ask.fc*ask.t);
            s_ask_d = s_ask_d - mean(s_ask_d);
            s_ask_d = lowpass(s_ask_d, ask.RB, ask.fs);
            metas_d = ones(1, ask.meta_num);
            for i = 0:ask.meta_num-1
                if s_ask_d(i*ask.meta_sampling+ask.meta_sampling/2)<0
                    s_ask_d(i*ask.meta_sampling+1:(i+1)*ask.meta_sampling) = 0;
                    metas_d(i+1) = 0;
                else
                    s_ask_d(i*ask.meta_sampling+1:(i+1)*ask.meta_sampling) = 1;
                end
            end
        end

        function [s_ask_d, metas_d] = ask_incoherent_demodulate(ask, s_ask_n) %非相干解调
            s_ask_d = bandpass(s_ask_n, [ask.fc-ask.RB, ask.fc+ask.RB], ask.fs);
            s_ask_d = abs(s_ask_d);
            s_ask_d = lowpass(s_ask_d, ask.RB, ask.fs); 
            metas_d = ones(1, ask.meta_num);
            for i = 0:ask.meta_num-1
                if s_ask_d(i*ask.meta_sampling+ask.meta_sampling/2)<0.5
                    s_ask_d(i*ask.meta_sampling+1:(i+1)*ask.meta_sampling) = 0;
                    metas_d(i+1) = 0;
                else
                    s_ask_d(i*ask.meta_sampling+1:(i+1)*ask.meta_sampling) = 1;
                end
            end
        end
        
        function show_signal(ask, s, s_ask, s_ask_n, s_ask_d, s_ask_d1) %绘制信号
            subplot(711);
            plot(ask.t,s);
            axis([0, 16, -0.2, 1.2]);
            title('基带信号');
        
            subplot(712);
            plot(ask.t,s_ask);
            axis([0, 16, -2, 2]);
            title('2ask调制信号');

            subplot(713);
            plot(ask.t,s_ask_n);
            axis([0, 16, -2, 2]);
            title('2ask调制信号加噪声');
        
            subplot(714);
            plot(ask.t,s_ask_d);
            axis([0, 16, -0.2, 1.2]);
            title('相干解调信号');
            
            subplot(715);
            plot(ask.t,s_ask_d1);
            axis([0, 16, -0.2, 1.2]);
            title('非相干解调信号');
        end
        
        function gen_data(ask, num, name) %生成训练集或验证集
            tic;
            features = zeros(9*num, ask.sum_sampling);
            labels = zeros(9*num, ask.meta_num);
            j = 0;
            for k = 1:0.5:5
                snr = 2*(k-1);
                for i = 1:num
                    m = ask.metas();
                    labels(i+j,:) = m;
                    s = ask.signal(m);
                    s_ask_n = ask.ask_modulate_noise(s, snr);
                    features(i+j,:) = s_ask_n; 
                end
                j = j + num;
            end   
            save(strcat(name,'/labels.mat'),"labels");
            save(strcat(name,'/features.mat'),"features");
            toc;
        end
        
        function gen_test_data(ask, num) %生成测试集
            tic;
            for k = 1:0.5:5
                snr = 2*(k-1);
                labels = zeros(num,ask.meta_num);
                features = zeros(num, ask.sum_sampling);
                for i = 1:num
                    m = ask.metas();
                    labels(i,:) = m;
                    s = ask.signal(m);
                    s_ask_n = ask.ask_modulate_noise(s, snr);
                    features(i,:) = s_ask_n;
                end
                save(strcat('test_data/labels_snr',num2str(snr),'.mat'),"labels");
                save(strcat('test_data/features_snr',num2str(snr),'.mat'),"features");
            end
            toc;
        end

        function [error_rate_v, error_rate_v1] = test(ask) %测试相干和非相干解调
            namelist_features = dir("test_data\features_snr*.mat");
            namelist_labels = dir("test_data\labels_snr*.mat");
            error_rate_v = zeros(1,length(namelist_labels));
            error_rate_v1 = zeros(1,length(namelist_labels));
            num = length(namelist_labels);
            for k = 1:num
                tic;
                test_data_features = load(namelist_features(k).name);
                test_data_features = cell2mat(struct2cell(test_data_features));
                test_data_labels = load(namelist_labels(k).name);
                test_data_labels = cell2mat(struct2cell(test_data_labels));
                snr = k-1;
                fprintf('snr: %d', snr);
                error_num = [0,0];
                for i = 1:size(test_data_labels,1)
                    s_ask_n = test_data_features(i,:);
                    m = test_data_labels(i,:);
                    [~, metas_d] = ask.ask_coherent_demodulate(s_ask_n);
                    [~, metas_d1] = ask.ask_incoherent_demodulate(s_ask_n);
                    error_num(1) = error_num(1) + numel(m,m~=metas_d);
                    error_num(2) = error_num(2) + numel(m,m~=metas_d1);
                end
                totalNum = num*16;
                error_rate_v(k) = error_num(1)/totalNum;
                error_rate_v1(k) = error_num(2)/totalNum;
                fprintf(' error rate: coherent_demodulate: %.4f incoherent_demodulate: %.4f\n', error_rate_v(k),error_rate_v1(k));
                toc;
            end
            save test_data/error_rate_v.mat error_rate_v;
            save test_data/error_rate_v1.mat error_rate_v1;
        end
                    

    end
end


