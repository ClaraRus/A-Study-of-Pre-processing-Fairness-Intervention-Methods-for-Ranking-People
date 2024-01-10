# code for estimating the causal model
#if(!require(devtools)) install.packages("devtools")
#devtools::install_github("cardiomoon/processR")
#http://www.regorz-statistik.de/en/mediation_process_for_r.html#download

library(mediation)
library(mma)
library(bruceR)

save_mediation_results <- function (res, group_list, out_path){
    df_total <- res$bin.result$results$total.effect
    df_direct <- res$bin.result$results$direct.effect
    df_indirect <- res$bin.result$results$indirect.effect

    cols = list()
    for(i in 1:length(names(df_indirect)))
    {
        cols[i] <- strsplit(names(df_indirect)[i], 'pred')[[1]][2]
    }


    for(i in 1:length(group_list))
    {
        if(group_list[i] %in% cols)
        {
            col <- paste0("pred", group_list[i], sep='')
        }
        else{
            col <- "pred"
        }

        file_name <- paste0(group_list[i], "_med.csv")
        Metric <- c("Indirect Effect", "Direct Effect", "Total Effect")
        Estimate <- c(df_indirect[[col]][, "y1.all"]['est'], df_direct[, paste("y1.", col, sep='')]['est'], df_total[, paste("y1.", col, sep='')]['est'])
        df <- data.frame(Metric, Estimate)
        write.csv(df, file=paste(out_path, file_name, sep='/'))
    }
}

get_mediators <-function (out_path, data_i){
    if(grepl('XING',out_path,  fixed = TRUE)) {
        idx = 4
    }
    else {
        idx = 6
    }

    if('Unnamed: 0' == colnames(data_i)[[1]]) {
            stop = length(colnames(data_i))-2
            cols = colnames(data_i[idx:stop])
    }
    else {
           start = idx-1
           stop = length(colnames(data_i))-2
           cols = colnames(data_i[start:stop])
    }
    return(cols)
}


estimate_causal_model <- function (data_i, IV, DV, MED, control, out_path){
    dir.create(out_path, showWarnings = TRUE)

    # Create Causal Model
    data_i[, IV] <- as.factor(data_i[, IV])

    group_list<- unique(data_i[[IV]])
    group_list<-group_list[group_list != control]

    print("Mediation...")
    # https://www.rdocumentation.org/packages/mma/versions/10.6-1/topics/mma
    med_cols = get_mediators(out_path, data_i)

    check_med <- data.org(x=data_i[med_cols],y=data_i[, DV],pred=data_i[, IV], mediator=med_cols, predref=control)

    if(check_med == 'no mediators found' || is.null(check_med$bin.results$contm)) {
        print("No mediators found!")
        Mediators = c('NULL')
        df <- data.frame(Mediators)
        file_name = "identified_mediators.csv"
        write.csv(df, file=paste(out_path, file_name, sep='/'))

        print("Test Effect of IV to DV")
        model.str = paste0(DV, "~", IV, '-1')
        form1 = as.formula(model.str)
        model.dv_iv <- lm(form1, data = data_i)
        write.csv(data.frame(summary(model.dv_iv)$coefficients), file=paste(out_path, paste0(model.str, '.csv', sep=''), sep='/'))

    }
    else{
        count_values =  table(data_i[, IV])
        min_count_values = min(count_values)


        file_name = "count_values.csv"
        write.csv(count_values, file=paste(out_path, file_name, sep='/'))
        med_i<-med(data=check_med)

        capture.output(print(med_i), file=paste(out_path, "med_output.txt", sep='/'))
        mediators <-data.org(x=data_i[med_cols],y=data_i[, DV],pred=data_i[, IV], mediator=med_cols, predref=control)
        sum_mediators = summary(mediators)
        Mediators = sum_mediators$mediator
        df <- data.frame(Mediators)
        file_name = "identified_mediators.csv"
        write.csv(df, file=paste(out_path, file_name, sep='/'))

        for(i in 1:length(sum_mediators$mediator))
        {
            model.str = paste0(sum_mediators$mediator[[i]], "~", IV, '-1')
            form2 = as.formula(model.str)
            model.iv_med <- lm(form2, data=data_i)

            write.csv(data.frame(summary(model.iv_med)$coefficients), file=paste(out_path, paste0(model.str, '.csv', sep=''), sep='/'))
        }
        }
    print("Done causal model estimation")
}