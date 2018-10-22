function main()


    SVM = SVM_for_face_recognition(false,'poly');
    tic
    accuracy = SVM.Five_fold_Cross_validation()
    toc
    SVM.save_SVM();
end