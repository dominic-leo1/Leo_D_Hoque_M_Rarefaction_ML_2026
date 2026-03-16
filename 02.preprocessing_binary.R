###import library
lapply(c("ggplot2", "qiime2R","phyloseq","file2meco", "microeco", "dplyr", "VennDiagram",
         "microbiome", "ggpubr", "scales", "patchwork", "cowplot", "mikropml", "readr",
         "tidyverse", "treemapify"), require, character.only = TRUE)


###load data
# physeq <- readRDS("sciuti.rds")

###Machine learning feature selection on true cases (samples without antibiotic use=======
##on combined dataset ====
#keeping samples without antibiotic use
# physeq1 <- subset_samples(physeq, status2 %in% c("A", "S"))

paper_id <- "1_PRJEB33711"
rarefaction_amount <- red_line # change
nonrarefied <- physeq1
# Increase max size to 10 GB
options(future.globals.maxSize = 10 * 1024^3)  # 10GB

########################## rarefied ############################
#rarefaction
dir.create("prepro_rarefied_data")

set.seed(123) 
ps.rarefied = rarefy_even_depth(physeq1, rngseed=1, sample.size=rarefaction_amount, replace=F)

#metadata
physeq.metadata <- ps.rarefied@sam_data
physeq.metadata$SampleID <- rownames(physeq.metadata)
#physeq.metadata <- physeq.metadata[,c(11,1:10)]

### with asv/otu feature========================
#otu table
otu.table <- as.data.frame(t(ps.rarefied@otu_table))
otu.table <- cbind(ps.rarefied@sam_data, otu.table)
#otu.table <- otu.table[-c(1:4,6:10)]

##feature selection========================= 
set.seed(123)
meco.physeq.rarefied <- phyloseq2meco(ps.rarefied)
meco.physeq.rarefied$cal_abund()

meco.physeq1 <- meco.physeq.rarefied$merge_samples("sample_type")
#meco.physeq1 <- meco.physeq.rarefied$merge_samples(use_group = "sample_type")
tiff("plot/110.rarefied_venn.tiff", units="in", width=7, height=4, res=600, compression = 'lzw')
t1 <- trans_venn$new(meco.physeq1, ratio = "numratio")
t1$plot_venn(color = c("#1F77B4FF","#D62728FF","#228B22FF"))
dev.off()

# shared ASVs
list_ctrl_dis <- as.data.frame(t1[["data_details"]]$`VAP&control`) # change
list_ctrl_dis <- list_ctrl_dis %>% filter(t1[["data_details"]]$`VAP&control` != '') # change
list_ctrl_dis <- list_ctrl_dis$'t1[["data_details"]]$`VAP&control`' # change

unique.otu.table <- otu.table %>% select(-c(list_ctrl_dis)) # change

#preproc with zv from unique set
otu.table.preproc <- preprocess_data(unique.otu.table, "sample_type", method = c("center", "scale"),
                                     remove_var = 'zv', collapse_corr_feats = FALSE,
                                     to_numeric = TRUE,  group_neg_corr = FALSE,
                                     prefilter_threshold = 1)$dat_transformed

unique.zv.list <- colnames(otu.table.preproc[-c(1)])
unique.zv.list <- gsub("`", "", unique.zv.list)
unique.zv.list <- gsub("_1", "", unique.zv.list)

#preproc with nzv from whole set
otu.table.preproc <- preprocess_data(otu.table, "sample_type", method = c("center", "scale"),
                                     remove_var = 'nzv', collapse_corr_feats = FALSE,
                                     to_numeric = TRUE,  group_neg_corr = FALSE,
                                     prefilter_threshold = 1)$dat_transformed

nzv.list <- colnames(otu.table.preproc[-c(1)])
unique.and.nzv.list <- unique(c(unique.zv.list, nzv.list))
unique.nzv.otu.table <- otu.table %>% select(c(unique.and.nzv.list))
unique.nzv.otu.table <- cbind(otu.table$sample_type, unique.nzv.otu.table)
unique.nzv.otu.table <- unique.nzv.otu.table %>% rename('type' = 'otu.table$sample_type')

#preproc with zv from merged of unique and whole set
otu.table.preproc <- preprocess_data(unique.nzv.otu.table, "type", method = c("center", "scale"),
                                     remove_var = 'zv', collapse_corr_feats = FALSE,
                                     to_numeric = TRUE,  group_neg_corr = FALSE,
                                     prefilter_threshold = 1)$dat_transformed

write_tsv(otu.table.preproc, file = paste0("prepro_rarefied_data/", paper_id, "_rarefied_asv.tsv"))


### with taxa feature========================
#check with unique and nzv taxa
meco.physeq <- phyloseq2meco(ps.rarefied)
meco.physeq$tidy_dataset()
meco.physeq$cal_abund()
d1 <- trans_classifier$new(dataset = meco.physeq, y.response = "sample_type", x.predictors = "All")
taxa.table <- d1$data_feature
taxa.table <- cbind(physeq.metadata[, 1, drop = FALSE], taxa.table[, -1])
# rownames(taxa.table) <- NULL
colnames(taxa.table) <- gsub("[[:punct:]]", "_", colnames(taxa.table))
colnames(taxa.table) <- gsub("k__Bacteria_p", "p", colnames(taxa.table))
colnames(taxa.table) <- gsub("__", "_", colnames(taxa.table))

taxa_uniq_list <- taxa.table %>% pivot_longer(cols = -c(1), names_to = "taxa", values_to = "abundance")
taxa_uniq_list <- taxa_uniq_list %>% filter(abundance > 0)
taxa_list_ctrl <- taxa_uniq_list %>% filter(sample_type == "control")
taxa_list_ctrl <- unique(taxa_list_ctrl$taxa)
taxa_list_dis <- taxa_uniq_list %>% filter(sample_type == "VAP") # change
taxa_list_dis <- unique(taxa_list_dis$taxa) # change

taxa_shared <- intersect(taxa_list_ctrl, taxa_list_dis) # change
taxa_uniq_list_ctrl <- setdiff(taxa_list_ctrl, taxa_shared)
taxa_uniq_list_pp <- setdiff(taxa_list_dis, taxa_shared) # change
unique.taxa.table <- taxa.table %>% select(-c(taxa_shared))

#preproc with zv from unique set
taxa.table.preproc <- preprocess_data(unique.taxa.table, "sample_type", method = c("center", "scale"),
                                      remove_var = 'zv', collapse_corr_feats = FALSE,
                                      to_numeric = TRUE,  group_neg_corr = FALSE,
                                      prefilter_threshold = 1)$dat_transformed

unique.zv.list <- colnames(taxa.table.preproc[-c(1)])
unique.zv.list <- gsub("`", "", unique.zv.list)

#preproc with nzv from whole set
taxa.table.preproc <- preprocess_data(taxa.table, "sample_type", method = c("center", "scale"),
                                      remove_var = "nzv", collapse_corr_feats = FALSE,
                                      to_numeric = TRUE,  group_neg_corr = FALSE,
                                      prefilter_threshold = 1)$dat_transformed

nzv.list <- colnames(taxa.table.preproc[-c(1)])
unique.and.nzv.list <- unique(c(unique.zv.list, nzv.list))
unique.nzv.taxa.table <- taxa.table %>% select(c(unique.and.nzv.list))
unique.nzv.taxa.table <- cbind(taxa.table$sample_type, unique.nzv.taxa.table)
unique.nzv.taxa.table <- unique.nzv.taxa.table %>% rename('type' = 'taxa.table$sample_type')

#preproc with zv from merged of unique and whole set
taxa.table.preproc <- preprocess_data(unique.nzv.taxa.table, "type", method = c("center", "scale"),
                                      remove_var = 'zv', collapse_corr_feats = FALSE,
                                      to_numeric = TRUE,  group_neg_corr = FALSE,
                                      prefilter_threshold = 1)$dat_transformed


write_tsv(taxa.table.preproc, file = paste0("prepro_rarefied_data/", paper_id, "_rarefied_taxa.tsv"))



########################## nonrarefied ############################
# no rarefaction
#metadata
physeq.metadata <- nonrarefied@sam_data
physeq.metadata$SampleID <- rownames(physeq.metadata)
#physeq.metadata <- physeq.metadata[,c(11,1:10)]

### with asv/otu feature========================
#otu table
otu.table <- as.data.frame(t(nonrarefied@otu_table))
otu.table <- cbind(nonrarefied@sam_data, otu.table)
#otu.table <- otu.table[-c(1:4,6:10)]

##feature selection========================= 
set.seed(123)
meco.physeq.nonrarefied <- phyloseq2meco(nonrarefied)
meco.physeq.nonrarefied$cal_abund()

meco.physeq1 <- meco.physeq.nonrarefied$merge_samples("sample_type")
#meco.physeq1 <- meco.physeq.rarefied$merge_samples(use_group = "sample_type")
tiff("plot/110.nonrarefied_venn.tiff", units="in", width=7, height=4, res=600, compression = 'lzw')
t1 <- trans_venn$new(meco.physeq1, ratio = "numratio")
t1$plot_venn(color = c("#1F77B4FF","#D62728FF","#228B22FF"))
dev.off()

# shared ASVs
taxa_list_dis <- as.data.frame(t1[["data_details"]]$`VAP&control`) # change
taxa_list_dis <- taxa_list_dis %>% filter(t1[["data_details"]]$`VAP&control` != '') # change
taxa_list_dis <- taxa_list_dis$'t1[["data_details"]]$`VAP&control`' # change

unique.otu.table <- otu.table %>% select(-c(taxa_list_dis)) # change

#preproc with zv from unique set
otu.table.preproc <- preprocess_data(unique.otu.table, "sample_type", method = c("center", "scale"),
                                     remove_var = 'zv', collapse_corr_feats = FALSE,
                                     to_numeric = TRUE,  group_neg_corr = FALSE,
                                     prefilter_threshold = 1)$dat_transformed

unique.zv.list <- colnames(otu.table.preproc[-c(1)])
unique.zv.list <- gsub("`", "", unique.zv.list)
unique.zv.list <- gsub("_1", "", unique.zv.list)

#preproc with nzv from whole set
otu.table.preproc <- preprocess_data(otu.table, "sample_type", method = c("center", "scale"),
                                     remove_var = 'nzv', collapse_corr_feats = FALSE,
                                     to_numeric = TRUE,  group_neg_corr = FALSE,
                                     prefilter_threshold = 1)$dat_transformed

nzv.list <- colnames(otu.table.preproc[-c(1)])
unique.and.nzv.list <- unique(c(unique.zv.list, nzv.list))
unique.nzv.otu.table <- otu.table %>% select(c(unique.and.nzv.list))
unique.nzv.otu.table <- cbind(otu.table$sample_type, unique.nzv.otu.table)
unique.nzv.otu.table <- unique.nzv.otu.table %>% rename('type' = 'otu.table$sample_type')

#preproc with zv from merged of unique and whole set
otu.table.preproc <- preprocess_data(unique.nzv.otu.table, "type", method = c("center", "scale"),
                                     remove_var = 'zv', collapse_corr_feats = FALSE,
                                     to_numeric = TRUE,  group_neg_corr = FALSE,
                                     prefilter_threshold = 1)$dat_transformed

write_tsv(otu.table.preproc, file = paste0("prepro_rarefied_data/", paper_id, "_nonrarefied_asv.tsv"))


### with taxa feature========================
#check with unique and nzv taxa
meco.physeq <- phyloseq2meco(nonrarefied)
meco.physeq$tidy_dataset()
meco.physeq$cal_abund()
d1 <- trans_classifier$new(dataset = meco.physeq, y.response = "sample_type", x.predictors = "All")
taxa.table <- d1$data_feature
taxa.table <- cbind(physeq.metadata[, 1, drop = FALSE], taxa.table[, -1])
# rownames(taxa.table) <- NULL
colnames(taxa.table) <- gsub("[[:punct:]]", "_", colnames(taxa.table))
colnames(taxa.table) <- gsub("k__Bacteria_p", "p", colnames(taxa.table))
colnames(taxa.table) <- gsub("__", "_", colnames(taxa.table))

taxa_uniq_list <- taxa.table %>% pivot_longer(cols = -c(1), names_to = "taxa", values_to = "abundance")
taxa_uniq_list <- taxa_uniq_list %>% filter(abundance > 0)
taxa_list_ctrl <- taxa_uniq_list %>% filter(sample_type == "control")
taxa_list_ctrl <- unique(taxa_list_ctrl$taxa)
taxa_list_dis <- taxa_uniq_list %>% filter(sample_type == "VAP") # change
taxa_list_dis <- unique(taxa_list_dis$taxa) # change

taxa_shared <- intersect(taxa_list_ctrl, taxa_list_dis) # change
taxa_uniq_list_ctrl <- setdiff(taxa_list_ctrl, taxa_shared)
taxa_uniq_list_dis <- setdiff(taxa_list_dis, taxa_shared) # change
unique.taxa.table <- taxa.table %>% select(-c(taxa_shared))

#preproc with zv from unique set
taxa.table.preproc <- preprocess_data(unique.taxa.table, "sample_type", method = c("center", "scale"),
                                      remove_var = 'zv', collapse_corr_feats = FALSE,
                                      to_numeric = TRUE,  group_neg_corr = FALSE,
                                      prefilter_threshold = 1)$dat_transformed

unique.zv.list <- colnames(taxa.table.preproc[-c(1)])
unique.zv.list <- gsub("`", "", unique.zv.list)

#preproc with nzv from whole set
taxa.table.preproc <- preprocess_data(taxa.table, "sample_type", method = c("center", "scale"),
                                      remove_var = "nzv", collapse_corr_feats = FALSE,
                                      to_numeric = TRUE,  group_neg_corr = FALSE,
                                      prefilter_threshold = 1)$dat_transformed

nzv.list <- colnames(taxa.table.preproc[-c(1)])
unique.and.nzv.list <- unique(c(unique.zv.list, nzv.list))
unique.nzv.taxa.table <- taxa.table %>% select(c(unique.and.nzv.list))
unique.nzv.taxa.table <- cbind(taxa.table$sample_type, unique.nzv.taxa.table)
unique.nzv.taxa.table <- unique.nzv.taxa.table %>% rename('type' = 'taxa.table$sample_type')

#preproc with zv from merged of unique and whole set
taxa.table.preproc <- preprocess_data(unique.nzv.taxa.table, "type", method = c("center", "scale"),
                                      remove_var = 'zv', collapse_corr_feats = FALSE,
                                      to_numeric = TRUE,  group_neg_corr = FALSE,
                                      prefilter_threshold = 1)$dat_transformed


write_tsv(taxa.table.preproc, file = paste0("prepro_rarefied_data/", paper_id, "_nonrarefied_taxa.tsv")) # change




########################## end ############################
