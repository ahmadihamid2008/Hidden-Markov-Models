clear
clc
close all
%% load data
lqseqs;
qs_len=length(qseqs);

%% parameters
type = 'discrete'; % type of HMM models
nc = 5; %number of clusters
O = 5; %number of observations
Q = 3; %number of HMM hidden states
max_k_means_iter = 50;

%% training
scores=zeros(nc,qs_len); %initialize score of each sequence by each model
%initialize clusters
clusters=zeros(1,qs_len);
for i=1:qs_len
    clusters(1,i)=randi(nc); 
end

%initialize HMM models
for m=1:nc
    models{m}.prior = normalise(rand(Q,1));
    models{m}.transmat = mk_stochastic(rand(Q,Q));
    models{m}.obsmat = mk_stochastic(rand(Q,O));
end

%plot
sum_of_log_lhood =[];
sum_of_max_scores = [];
num_of_changed_clusters = [];
figure;
h_sum_of_max_scores = plot(zeros(1,1)); title('sum_of_max_scores');
figure;
h_sum_of_log_lhood = plot(zeros(1,1)); title('sum_of_log_lhood');
figure;
h_num_of_changed_clusters = plot(zeros(1,1)); title('num_of_changed_clusters');

iter=1; %k_means iteration
while iter <= max_k_means_iter
    sum_of_log_lhood = [sum_of_log_lhood 0];
    iter=iter+1;
    %train each model using thats seqs
    for m=1:nc
        seqs={};
        for sq =  1:qs_len
            if clusters(sq)==m
                seqs{end+1,1}=qseqs{sq};
            end
        end
                
        [LL,p,t,o] = dhmm_em(seqs, models{m}.prior, models{m}.transmat, models{m}.obsmat, 'max_iter', 30);
        train_rate = (iter/max_k_means_iter).^2; % 0 to 1
        models{m}.prior = train_rate*models{m}.prior + (1-train_rate)*p;
        models{m}.transmat = train_rate*models{m}.transmat + (1-train_rate)*t;
        models{m}.obsmat = train_rate*models{m}.obsmat + (1-train_rate)*o;
        
        %[models{m},logl{m}]=hmmFit(seqs, ns, type);
        sum_of_log_lhood(end)=sum_of_log_lhood(end)+LL(end);
    end
    %assign clusters
    for sq = 1:length(qseqs)
        for m = 1:nc
            scores(m,sq)=dhmm_logprob(qseqs(sq),models{m}.prior, models{m}.transmat, models{m}.obsmat);
        end
    end
    [maxp,clusters(iter,:)]=max(scores);
    sum_of_max_scores = [sum_of_max_scores sum(maxp)];
    num_of_changed_clusters=[num_of_changed_clusters sum(clusters(iter-1,:)~=clusters(iter,:))];
    
    disp([iter num_of_changed_clusters(end) sum_of_log_lhood(end) sum_of_max_scores(end)]);
    set(h_sum_of_max_scores,'YData',sum_of_max_scores);
    set(h_sum_of_log_lhood,'YData',sum_of_log_lhood);
    set(h_num_of_changed_clusters,'YData',num_of_changed_clusters);
    drawnow;
end