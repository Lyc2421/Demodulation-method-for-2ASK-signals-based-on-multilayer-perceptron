ask = ASK(16,64,1,5);
%% 打印和绘制各种信号
m = ask.metas();
s = ask.signal(m);
s_ask = ask.ask_modulate(s);
s_ask_n = ask.ask_modulate_noise(s, 10);
[s_ask_d, metas_d] = ask.ask_coherent_demodulate(s_ask_n);
[s_ask_d1, metas_d1] = ask.ask_incoherent_demodulate(s_ask_n);

disp(m);
disp(metas_d);
disp(metas_d1);
ask.show_signal(s, s_ask, s_ask_n, s_ask_d, s_ask_d1);
%% 生成测试集并保存
ask.gen_test_data(100);
%% 测试相干和非相干解调，花费6.5分钟
[v,v1] = ask.test();
disp(v);
disp(v1);
%% 生成训练集和验证集并保存
ask.gen_data(300, 'train_data');
ask.gen_data(100, 'val_data');
