classdef SVM_for_face_recognition
    properties(Access = private)
        DataSet_path  %folder name of dataset(the folder need to be at the same folder of class.m)
        Classifers %[(w11 w12...w1n b1);(w21 w22...w2n b2)...;(wend1 wend2...wendn bend)];
                        %end = k(k-1)/2 
        Trained_tag
        Kernal_type
        Categories_nums%nums of categories
        Train_parameters
        Total_dataset% every row is [X Y];
        Train_dataset% every row is [X Y];
        Test_dataset%  every row is [X Y];
        Regular_img_size = [64 64];
        
    end
    methods(Access=public)
        function obj =  SVM_for_face_recognition(load_tag,Kernal_type)%load trained svm tag
            if(load_tag==true)
                if(exist('./SVM.mat','file')==2)
                    load('./SVM.mat','SVM');
                    obj = SVM;
                else
                    fprintf("file SVM.mat not exist\n");
                end
            else
                [obj,Categories_nums] = obj.load_dataset("att_faces");
                obj.Categories_nums = Categories_nums;
                obj.Kernal_type = Kernal_type;
                obj.Train_dataset = [];
                obj.Test_dataset = [];
            end
        end
        function [obj, Categories_nums] = load_dataset(obj,foldername)
            if(exist(foldername,'dir')==7)
                Datas_tmp = [];
                %subfolder = "s1";
                addpath(foldername);
                index1 = 1;
                while(true)
                    subfolder = foldername+'/s'+int2str(index1);
                    if(exist(subfolder,'dir')~=7)
                        break;
                    end
                    index2=1;
                    while(true)
                        filename = subfolder+'/'+int2str(index2)+'.pgm';
                        if(exist(filename,'file')~=2)
                            break;
                        end
                        R = imread(filename);
                        Features = obj.extractF(R);
                        %Features = double(reshape(Features,1,size(Features,1)*size(Features,2)));
                        Features = [Features index1];
                        Datas_tmp = [Datas_tmp; Features];
                        index2 = index2+1;
                    end
                    index1 = index1+1;
                end
                Categories_nums = index1-1;
                obj.Total_dataset = Datas_tmp;
            else
                fprintf("dataset folder:%s not exist\n",foldername);
                pause();
            end
            fprintf("load dataset:%s over\n",foldername);
        end
        function obj = fix(obj)%train svm
            if(isempty(obj.Train_dataset))
                obj.Train_dataset = obj.Total_dataset;
            end
            Indexes = {ones(obj.Categories_nums*(obj.Categories_nums-1)/2,1)};
            k=1;
            for i = 1:obj.Categories_nums
                for j = i+1:obj.Categories_nums
                    Indexes{k} = [i j];
                    k = k+1;
                end
            end
            %callculate all classifiers
            if( strcmp(obj.Kernal_type, 'liner')==1)%same
                Support_vectors = cell2mat(cellfun(@(x) obj.optim_with_quadprog(x(1),x(2)),Indexes,'UniformOutput',false));
                Support_vectors = Support_vectors';
                %save support_vectors with [class1 class2] in the tail.
                obj.Classifers = mat2cell([Support_vectors cell2mat(Indexes')],ones(1,k-1));
            else
                Support_vectors = cellfun(@(x) obj.optim_alpha_with_quadprog(x(1),x(2)),Indexes,'UniformOutput',false);
                obj.Classifers = Support_vectors';
            end
            
            fprintf("Train over\n");
        end
        function categories_name = predict(obj,Image_input,Features)
            if(isempty(Features))
                Features = obj.extractF(Image_input);
            end
            vote = zeros(obj.Categories_nums*(obj.Categories_nums-1)/2,1);
            for i = 1:obj.Categories_nums
                for j = i+1:obj.Categories_nums
                    class_name = obj.predict_two(Features,i,j);
                    if(class_name==1)
                        vote(i) = vote(i)+1;
                    else
                        vote(j) = vote(j)+1;
                    end
                    %fprintf("i:%d,j:%d  over\n",i,j);
                end
            end
            [~,categories_name] = max(vote);
%             Indexes = {ones(obj.Categories_nums*(obj.Categories_nums-1)/2,1)};
%             k=1;
%             for i = 1:obj.Categories_nums
%                 for j = i+1:obj.Categories_nums
%                     Indexes{k} = [i j];
%                     k = k+1;
%                 end
%             end
%             Result = cell2mat(cellfun(@(x)...
%                         obj.predict_two(Features,x(1),x(2))*x(1)+...
%                             ~obj.predict_two(Features,x(1),x(2))*x(2)...
%                         ,Indexes,'UniformOutput',false));
%             table=tabulate(Result);
%             [F,~]=max(table(:,2));
%             I=table(:,2)==F;
%             categories_name = table(I,1);
%             
%             
            
            
            fprintf("predict over\n");
        end
        function accuracy = Five_fold_Cross_validation(obj)
            Dataset= mat2cell(obj.Total_dataset,ones(1,size(obj.Total_dataset,1)));
            index = (1:1:size(obj.Total_dataset,1))';
            %shuffle the index randomly
            index=index(randperm(length(index)));
            groupsize = size(index,1)/5;
            %% calculate all five gourps of indexes
            %1
            index1_test = index(1:groupsize);
            index1_train = index;index1_train(1:groupsize) = [];
            %2
            index2_test = index(1*groupsize+1:2*groupsize);
            index2_train = index;index2_train(1*groupsize+1:2*groupsize) = [];
            %3
            index3_test = index(2*groupsize+1:3*groupsize);
            index3_train = index;index3_train(2*groupsize+1:3*groupsize) = [];
            %4
            index4_test = index(3*groupsize+1:4*groupsize);
            index4_train = index;index4_train(3*groupsize+1:4*groupsize) = [];
            %5
            index5_test = index(4*groupsize+1:end);
            index5_train = index;index5_train(4*groupsize+1:end) = [];
            %% devide the dataset into train and test based on the indexes
            Train1 = cell2mat(Dataset(index1_train));
            Train2 = cell2mat(Dataset(index2_train));
            Train3 = cell2mat(Dataset(index3_train));
            Train4 = cell2mat(Dataset(index4_train));
            Train5 = cell2mat(Dataset(index5_train));
            Test1 = cell2mat(Dataset(index1_test));
            Test2 = cell2mat(Dataset(index2_test));
            Test3 = cell2mat(Dataset(index3_test));
            Test4 = cell2mat(Dataset(index4_test));
            Test5 = cell2mat(Dataset(index5_test));
            %% validata on these five groups
            obj.Train_dataset = Train1;
            obj.Test_dataset =  Test1;
            accuracy1 = obj.validate_on_current_data();
            fprintf("20%/n");
            obj.Train_dataset = Train2;
            obj.Test_dataset =  Test2;
            accuracy2 = obj.validate_on_current_data();
            fprintf("40%/n");
            obj.Train_dataset = Train3;
            obj.Test_dataset =  Test3;
            accuracy3 = obj.validate_on_current_data();
            fprintf("60%/n");
            obj.Train_dataset = Train4;
            obj.Test_dataset =  Test4;
            accuracy4 = obj.validate_on_current_data();
            fprintf("80%/n");
            obj.Train_dataset = Train5;
            obj.Test_dataset =  Test5;
            accuracy5 = obj.validate_on_current_data();
            fprintf("100%/n");
            accuracy = mean([accuracy1;accuracy2;accuracy3;accuracy4;accuracy5]);
        end
        function accuracy = validate_on_current_data(obj)
            obj = obj.fix();
            
            result = cell2mat(cellfun(@(x) obj.predict([],x(1:end-1))==x(end),...
                           mat2cell(obj.Test_dataset,ones(1,size(obj.Test_dataset,1))),...
                           'UniformOutput',false));
            accuracy = sum(result,1)/size(result,1);
        end
        function save_SVM(obj)
            SVM =obj;
            save('SVM.mat','SVM');
        end
        function Features = extractF(obj,R)
            R=imresize(R,obj.Regular_img_size);
            %R = double(reshape(R,1,[]));
%             
%             axis_devide = 4;%devide img into 4*4 parts
%             %%devide raw image into axis_devide*axis_devide
%             %%parts then extract LBP featrues  from them
%             R2 = mat2cell(R,...
%                                 repmat(size(R,1)/axis_devide,1,axis_devide),...%row
%                                 repmat(size(R,2)/axis_devide,1,axis_devide));%collum
%             E = cellfun(@(x) extractLBPFeatures(x,'CellSize',[4 4]),R2,'UniformOutput',false);
%             % range the featrue in featrues*(n*n)
%             E = cell2mat(reshape(E,size(E,1)*size(E,2),1));
%             %stanterlize through collum;
%             %input n-by-p,output p-by-p
%             [COEFF,SCORE,latent]  = pca(zscore(E),'Economy',true);
%             [M,I] = max(latent);
%             x = extractLBPFeatures(R,'CellSize',[16 16]);
%             Features = x*COEFF;
            Features = extractLBPFeatures(R);
            %Features = Features*COEFF;
            Features = reshape(Features,1,size(Features,1)*size(Features,2));
            %Features = eig((Features+Features')/2)';
            Features = double(Features);
            if(~isreal(Features))
                fprintf("error in calcu Feature\n");
                pause();
            end
        end
    end
    methods(Access = private)
        function Support_vector = optim_with_quadprog(obj,class1,class2)
            %% codode data into a specific form for using quadprog
            Data_features = obj.calcu_data_features(class1,class2);
            A = -diag(Data_features.Y)*[Data_features.X ones(Data_features.l,1)];
            c = -ones(Data_features.l,1);
            H = eye(Data_features.n+1);H(end,end)=0;
            f = zeros(Data_features.n+1,1);
            %% optimize hyper-plane using quadprog
            %A: l*(attribu+1) H: attribus+1
            options = optimoptions('quadprog','Display','off');
            Aeq = [];
            beq = [];
            lb = [];
            ub = [];
            x0 = [];
            Support_vector = quadprog(H,f,A,c,Aeq,beq,lb,ub,x0,options);
            
        end
        function Support_vector = optim_alpha_with_quadprog(obj,class1,class2)
            %% codode data into a specific form for using quadprog
            Data_features = obj.calcu_data_features(class1,class2);
            %% optimize hyper-plane using quadprog
            options = optimoptions('quadprog','Display','off');
            % min  1/2*x'*H*x + f'*x
            % s.t. A*x <= b
            %      Aeq*x = beq
            %      lb <= x <= ub
            % X and Y need be 1*n
            X = Data_features.X';
            Y = Data_features.Y';
            dataNumber = size(X, 2);

            H = (Y' * Y) .* obj.calcl_k(X, X);
            f = -ones(dataNumber, 1);
            %-alpha<=0 means alpha>=0
            A = -eye(dataNumber);
            b = zeros(dataNumber, 1);
            Aeq = Y;
            beq = 0;
            lb = [];
            ub = [];
            x0 = [];
            alpha  = quadprog(H, f, A, b, Aeq, beq,lb,ub,x0,options);  

            % 
            epsilon = 1e-5; 
            indexSV = find(alpha > epsilon);

            b = mean(Y(indexSV) - (alpha(indexSV)' .* Y(indexSV)) * obj.calcl_k(X(:, indexSV), X(:, indexSV)));
            Support_vector.alpha = alpha(indexSV);
            Support_vector.b = b;
            Support_vector.class = [class1 class2];
            Support_vector.X = X(:, indexSV);
            Support_vector.Y = Y(:, indexSV);
        end
        function Data_features = calcu_data_features(obj,class1,class2)
%             Data_features = struct('X',[],...%data points 
%                                'Y',[],...%categories name
%                                'l',[],...%num of train data point
%                                'n',[]);%num of Attributes
            %get data of class1
            [index_i,~] = find(obj.Train_dataset(:,end)==class1);
            Data1 = obj.Train_dataset(index_i, :);
            Data1(:,end) = 1;%set class tag equals +1
            %get data of class2;
            [index_i,~] = find(obj.Train_dataset(:,end)==class2);
            Data2 = obj.Train_dataset(index_i, :);
            Data2(:,end) = -1;%set class tag equals -1
            Data_features.X = [Data1(:,1:end-1);Data2(:,1:end-1)];
            Data_features.Y = [Data1(:,end);Data2(:,end)];
            Data_features.l = size(Data_features.X,1);
            Data_features.n = size(Data_features.X,2);
        end
        function classifier = load_classifier(obj,class1,class2)
           %check which one saves the classifier for class1 and class2
           if( strcmp(obj.Kernal_type, 'liner')==1)%same
                indexes = cellfun(@(x) x(end-1)==class1&&x(end)==class2,obj.Classifers);
           else
               indexes = cellfun(@(x) x.class(1)==class1&&x.class(2)==class2,obj.Classifers);
           end
           classifier = obj.Classifers{indexes==true};
        end
        
        function class_name = predict_two(obj,Features,class1,class2)
             classifier = obj.load_classifier(class1,class2);
             if( strcmp(obj.Kernal_type, 'liner')==1)%same
                class_name = sign([Features 1] * classifier(1:end-2)');
             else
                class_name = sign(sum(bsxfun(@times, classifier.alpha .* classifier.Y', obj.calcl_k(classifier.X, Features'))) + classifier.b);
             end
             if(class_name==-1)
                 class_name = 0;
             end
        end
        function kernel = calcl_k(obj,X,Y)
            switch obj.Kernal_type
                case 'liner'
                    kernel = X' * Y;
                case 'poly'
                    kernel = (1 + X' * Y).^2;
                case 'gauss'
                    numberX = size(X,2);
                    numberY = size(Y,2);   
                    tmp = zeros(numberX, numberY);
                    for i = 1:numberX
                        for j = 1:numberY
                            tmp(i,j) = norm(X(:,i)-Y(:,j));
                        end
                    end        
                    kernel = exp(-0.5*(tmp).^2);        
                otherwise
                    kernel = 0;
            end

        end
    end
end
            
    