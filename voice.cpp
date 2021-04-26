// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4;  tab-width: 8; -*-
//
// Simple example showing how to do the standard 'hello, world' using embedded R
//
// Copyright (C) 2009 Dirk Eddelbuettel 
// Copyright (C) 2010 Dirk Eddelbuettel and Romain Francois
//
// GPL'ed 

#include <RInside.h>                    // for the embedded R via RInside

int main(int argc, char *argv[]) {

    RInside R(argc, argv);              // create an embedded R instance 

    std::string str = 
        "packages <- c(\"tuneR\", 'seewave', 'fftw', 'caTools', 'randomForest', 'warbleR', 'mice', 'e1071', 'rpart', 'rpart-plot', 'xgboost', 'e1071');"
        "packages";
        //"options(repos=structure(c(CRAN='https://mirrors.tuna.tsinghua.edu.cn/CRAN/')));"
        //"if (length(setdiff(packages, rownames(installed.packages()))) > 0) {;"
        //"install.packages(setdiff(packages, rownames(installed.packages())))";
        //"}";
    Rcpp::StringVector packages = R.parseEval(str);
    for (int i=0; i< packages.size(); i++) {           // show the result
            std::cout << "In C++ element " << i << " is " << packages[i] << std::endl;
        };

    std::string library =
        "library(tuneR);"
        "library(seewave);"
        "library(caTools);"
        "library(rpart);"
        "library(rpart.plot);"
        "library(randomForest);"
        //"library(warbleR);"
        //"library(mice);"
        "library(xgboost);"
        "library(e1071)";
    R.parseEvalQ(library);
    std::string specan3_txt =
            "specan3 <- function(X, bp = c(0,22), wl = 2048, threshold = 5, parallel = 1){;"
            "if(class(X) == \"data.frame\") {if(all(c(\"sound.files\", \"selec\", \"start\", \"end\") %in% colnames(X))){;"
            "start <- as.numeric(unlist(X$start));"
            "end <- as.numeric(unlist(X$end));"
            "sound.files <- as.character(unlist(X$sound.files));"
            "selec <- as.character(unlist(X$selec));"
            "} else stop(paste(paste(c(\"sound.files\", \"selec\", \"start\", \"end\")[!(c(\"sound.files\", \"selec\",\"start\", \"end\") %in% colnames(X))], collapse=\", \"), \"column(s) not found in data frame\"));"
            "} else  stop(\"X is not a data frame\");"
            "if(any(is.na(c(end, start)))) stop(\"NAs found in start and/or end\")  ;"
            "if(all(class(end) != \"numeric\" & class(start) != \"numeric\")) stop(\"'end' and 'selec' must be numeric\");"
            "if(any(end - start<0)) stop(paste(\"The start is higher than the end in\", length(which(end - start<0)), \"case(s)\"))  ;"
            "if(any(end - start>20)) stop(paste(length(which(end - start>20)), \"selection(s) longer than 20 sec\"))  ;"
            "options( show.error.messages = TRUE);"
            "if(!is.vector(bp)) stop(\"'bp' must be a numeric vector of length 2\") else{;"
            "if(!length(bp) == 2) stop(\"'bp' must be a numeric vector of length 2\")};"
            "fs <- list.files(path = getwd(), pattern = \".wav$\", ignore.case = TRUE);"
            "if(length(unique(sound.files[(sound.files %in% fs)])) != length(unique(sound.files))){ ;"
            "cat(paste(length(unique(sound.files))-length(unique(sound.files[(sound.files %in% fs)])),\".wav file(s) not found\"))};"
            "d <- which(sound.files %in% fs) ;"
            "if(length(d) == 0){;"
            "stop(\"The .wav files are not in the working directory\");"
            "}  else {;"
            "start <- start[d];"
            "end <- end[d];"
            "selec <- selec[d];"
            "sound.files <- sound.files[d];"
            "};"
            "if(!is.numeric(parallel)) stop(\"'parallel' must be a numeric vector of length 1\") ;"
            "if(any(!(parallel %% 1 == 0),parallel < 1)) stop(\"'parallel' should be a positive integer\");"
            "if(parallel > 1){ options(warn = -1);"
            "if(all(Sys.info()[1] == \"Windows\",requireNamespace(\"parallelsugar\", quietly = TRUE) == TRUE)){;"
            "lapp <- function(X, FUN) parallelsugar::mclapply(X, FUN, mc.cores = parallel)} else if(Sys.info()[1] == \"Windows\"){ ;"
            "cat(\"Windows users need to install the 'parallelsugar' package for parallel computing (you are not doing it now!)\");"
            "lapp <- pbapply::pblapply} else lapp <- function(X, FUN) parallel::mclapply(X, FUN, mc.cores = parallel)} else lapp <- pbapply::pblapply;"
            "options(warn = 0);"
            "if(parallel == 1) cat(\"Measuring acoustic parameters:\");"
            "x <- as.data.frame(lapp(1:length(start), function(i) { ;"
            "r <- tuneR::readWave(file.path(getwd(), sound.files[i]), from = start[i], to = end[i], units = \"seconds\") ;"
            "b<- bp ;"
            "if(b[2] > ceiling(r@samp.rate/2000) - 1) b[2] <- ceiling(r@samp.rate/2000) - 1 ;"
            "songspec <- seewave::spec(r, f = r@samp.rate, plot = FALSE);"
            "analysis <- seewave::specprop(songspec, f = r@samp.rate, flim = c(0, 280/1000), plot = FALSE);"
            "meanfreq <- analysis$mean/1000;"
            "sd <- analysis$sd/1000;"
            "median <- analysis$median/1000;"
            "Q25 <- analysis$Q25/1000;"
            "Q75 <- analysis$Q75/1000;"
            "IQR <- analysis$IQR/1000;"
            "skew <- analysis$skewness;"
            "kurt <- analysis$kurtosis;"
            "sp.ent <- analysis$sh;"
            "sfm <- analysis$sfm;"
            "mode <- analysis$mode/1000;"
            "centroid <- analysis$cent/1000;"
            "peakf <- 0;"
            "ff <- seewave::fund(r, f = r@samp.rate, ovlp = 50, threshold = threshold,fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = wl)[, 2];"
            "meanfun<-mean(ff, na.rm = T);"
            "minfun<-min(ff, na.rm = T);"
            "maxfun<-max(ff, na.rm = T);"
            "y <- seewave::dfreq(r, f = r@samp.rate, wl = wl, ylim=c(0, 280/1000), ovlp = 0, plot = F, threshold = threshold, bandpass = b * 1000, fftw = TRUE)[, 2];"
            "meandom <- mean(y, na.rm = TRUE);"
            "mindom <- min(y, na.rm = TRUE);"
            "maxdom <- max(y, na.rm = TRUE);"
            "dfrange <- (maxdom - mindom);"
            "duration <- (end[i] - start[i]);"
            "changes <- vector();"
            "for(j in which(!is.na(y))){;"
            "change <- abs(y[j] - y[j + 1]);"
            "changes <- append(changes, change);"
            "};"
            "if(mindom==maxdom) modindx<-0 else modindx <- mean(changes, na.rm = T)/dfrange;"
            "return(c(duration, meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx));" 
            "}));"
            "rownames(x) <- c(\"duration\", \"meanfreq\", \"sd\", \"median\", \"Q25\", \"Q75\", \"IQR\", \"skew\", \"kurt\", \"sp.ent\", \"sfm\",\"mode\", \"centroid\", \"peakf\", \"meanfun\", \"minfun\", \"maxfun\", \"meandom\", \"mindom\", \"maxdom\", \"dfrange\", \"modindx\");"
            "x <- data.frame(sound.files, selec, as.data.frame(t(x)));"
            "colnames(x)[1:2] <- c(\"sound.files\", \"selec\");"
            "rownames(x) <- c(1:nrow(x));" 
            
            "return(x)}";         
    R.parseEvalQ(specan3_txt);
    
    std::string processFolder_txt =
            "processFolder <- function(folderName) {;"
            "data <- data.frame();"
            "list <- list.files(folderName, \"\\\\.wav\");"
            "print(folderName,'的音频数量:',length(list));"
            "for (fileName in list) {;"
            "row <- data.frame(fileName, 0, 0, 20);"
            "data <- rbind(data, row);"
            "};"
            "names(data) <- c('sound.files', 'selec', 'start', 'end');"
            "setwd(folderName);"
            "acoustics <- specan3(data, parallel=1);"
            "setwd('..');"
            "print(folderName,'Done!');"
            "acoustics}";
            
    R.parseEvalQ(processFolder_txt);
    std::string gender_txt =
            "gender <- function(filePath) {;"
            "if (!exists('genderBoosted')) {;"
            "load('model.bin');"
            "};"
            "currentPath <- getwd();"
            "fileName <- basename(filePath);"
            "path <- dirname(filePath);"
            "setwd(path);"
            "data <- data.frame(fileName, 0, 0, 20);"
            "names(data) <- c('sound.files', 'selec', 'start', 'end');"
            "acoustics <- specan3(data, parallel=1);"
            "setwd(currentPath);"
            "predict(genderCombo, newdata=acoustics)}";
    R.parseEvalQ(gender_txt);
    /* 读取新数据保存在males females */
    std::string data_to_csv_txt =
            "print(getwd());"
             "males <- processFolder('male');"
            "females <- processFolder('female');"
            "males$label <- 1;"
            "females$label <- 2;"
            "data <- rbind(males, females);"
            "data$label <- factor(data$label, labels=c('male', 'female'));"
            "data$duration <- NULL;"
            "data$sound.files <- NULL;"
            "data$selec <- NULL;"
            "data$peakf <- NULL;"
            "data <- data[complete.cases(data),];"
            "write.csv(data, file='lecvoice.csv', sep=',', row.names=F)";

    R.parseEvalQ(data_to_csv_txt); 

    std::string dataload_txt = 
            "data <- read.csv('lecvoice.csv');"
            "data$label<-as.factor(data$label);"
            "train<-data;"
            "set.seed(777);"
            "spl <- sample.split(data$label, 0.7);"
            "train <- subset(data, spl == TRUE);"
            "test <- subset(data, spl == FALSE)";

    //R.parseEvalQ(dataload_txt); 

    std::string glm_txt =
            "genderLog <- glm(label ~ ., data=train, family='binomial');"
            "predictLog <- predict(genderLog, type='response');"
            "predictLog2 <- predict(genderLog, newdata=test, type='response');"
            "print(predictLog2)";
    //R.parseEvalQ(glm_txt); 

    






    std::cout << "success!"  << std::endl;



    //R.parseEvalQ("cat(txt)"); 
    exit(0);
}

