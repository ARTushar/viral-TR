library(gkmSVM)

indir <- "train_valid"
outdir <- "train_valid_cv"

posfn <- paste(indir, "pos_seq.fa", sep="/") #positive set (FASTA format)
negfn <- paste(indir, "neg_seq.fa", sep="/") #negative set (FASTA format)
testfn <- paste(indir, "test_seq.fa", sep="/") #test set (FASTA format)

kernelfn <- paste(outdir, "kernel.txt", sep="/") #kernel matrix
svmfnprfx <- paste(outdir, "svmtrain", sep="/") #SVM files
outfn <- paste(outdir, "output.txt", sep="/") #output scores for sequences in the test set

print("==================== Computing Kernel ====================")
gkmsvm_kernel(posfn, negfn, kernelfn); #computes kernel

# print("==================== Training SVM ====================")
# gkmsvm_train(kernelfn, posfn, negfn, svmfnprfx); #trains SVM

rocpdf <- paste(outdir, "ROC.pdf", sep="/")
cvpred <- paste(outdir, "cvpred.out", sep="/")
print("==================== CV Training SVM ====================")
gkmsvm_trainCV(kernelfn, posfn, negfn, svmfnprfx, outputPDFfn=rocpdf, outputCVpredfn=cvpred
	       nCV=10, nrepeat=5, C=c(1, 0.9, 0.8, 1.1));

print("==================== Scoring Test Sequences ====================")
gkmsvm_classify(testfn, svmfnprfx, outfn); #scores test sequences
