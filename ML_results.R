# Install if needed
# install.packages(c("tidyverse", "readr", "viridis", "patchwork"))

# Load
library(tidyverse)
library(readr)
library(viridis)
library(patchwork)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(ggpubr)
library(rstatix)

############## USAGE ##############
# This has been written to compare the effect of rarefaction on 11 different papers using taxa and asv data
# The analysis can be performed with and without outliers (filtering based on IQR for skewed data or z-score for normal data)



# ===============================
# Rarefaction Comparison Boxplots
# ===============================
# ---- Load data ----
data <- read.csv("ml_summary_means.csv", stringsAsFactors = FALSE)

# ---- Clean dataset labels ----
data <- data %>%
  mutate(
    Rarefaction = ifelse(grepl("nonrarefied", Dataset),
                         "Non-rarefied",
                         "Rarefied"),
    Feature_Type = ifelse(grepl("asv", Dataset),
                          "ASV",
                          "Taxa"),
    ML_Number = factor(ML_Number)
  )

# ---- Convert metrics to numeric ----
metrics <- c("AUC", "Sensitivity", "Specificity", "Balanced_Accuracy")
main_metrics <- c("AUC", "Balanced_Accuracy")
supplementary_metrics <- c("Sensitivity", "Specificity")

data[metrics] <- lapply(data[metrics], as.numeric)

# ---- Reshape to long format ----
long_data <- data %>%
  pivot_longer(cols = all_of(metrics),
               names_to = "Metric",
               values_to = "Value")

# WITHOUT OUTLIERS #
long_data_no_outliers <- long_data %>%
  group_by(Model, Metric, Rarefaction) %>%
  mutate(
    Q1 = quantile(Value, 0.25, na.rm = TRUE),
    Q3 = quantile(Value, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1
  ) %>%
  filter(
    Value >= Q1 - 1.5 * IQR,
    Value <= Q3 + 1.5 * IQR
  ) %>%
  ungroup() %>%
  select(-Q1, -Q3, -IQR)

# ----------------------- taxa --------------------------
long_data_taxa <- long_data_no_outliers %>% # CHANGE TO INCLUDE OR REMOVE OUTLIERS
  dplyr::filter(grepl("taxa$", Dataset))

# Create combined dataset
combined_data_taxa <- long_data_taxa %>%
  mutate(ML_Number = "All")   # Add new pseudo-paper

long_data_taxa_extended <- bind_rows(
  long_data_taxa,
  combined_data_taxa
)

long_data_taxa_extended$ML_Number <- as.character(long_data_taxa_extended$ML_Number)

long_data_taxa_extended$ML_Number <- factor(
  long_data_taxa_extended$ML_Number,
  levels = c(as.character(1:16), "All")
)

long_data_taxa_extended <- long_data_taxa_extended %>%
  filter(!is.na(ML_Number))

# Separate into main and supplementary metrics 
long_data_all_main_taxa <- long_data_taxa_extended %>%
  filter(Metric %in% main_metrics)

long_data_all_supp_taxa <- long_data_taxa_extended %>%
  filter(Metric %in% supplementary_metrics)

# main metrics
# Shapiro-Wilk per ML_Number × Metric × Rarefaction for normality
normality_results_main_taxa <- long_data_all_main_taxa %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) {  # check for zero variance
        NA
      } else {
        shapiro.test(Value)$p.value
      }
    } else {
      NA
    },
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
      shapiro_p > 0.05 ~ "Normal",
      shapiro_p <= 0.05 ~ "Not normal"
    )
  )

print(normality_results_main_taxa)

# Select only the columns you want in the CSV
normality_csv_taxa <- normality_results_main_taxa %>%
  select(ML_Number, Metric, Rarefaction, n, shapiro_p, normality)

# Export to CSV
write.csv(
  normality_csv_taxa,
  "normality_results_main_taxa.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)


# -------------------------
# Run paired Wilcoxon test for taxa
# -------------------------
wilcox_results_main_taxa <- long_data_all_main_taxa %>%
  group_by(ML_Number, Metric) %>%
  summarise(
    p_value = wilcox.test(
      Value[Rarefaction == "Rarefied"],
      Value[Rarefaction == "Non-rarefied"],
      paired = TRUE
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p.adj = p.adjust(p_value, method = "BH"),
    p.adj.signif = case_when(
      p.adj <= 0.001 ~ "***",
      p.adj <= 0.01  ~ "**",
      p.adj <= 0.05  ~ "*",
      TRUE           ~ "ns"
    ),
    group1 = "Non-rarefied",  # first box
    group2 = "Rarefied",      # second box
    # y.position slightly above max of this ML_Number × Metric
    y.position = long_data_all_main_taxa %>%
      filter(ML_Number == unique(ML_Number),
             Metric == unique(Metric)) %>%
      summarise(max_val = max(Value)) %>%
      pull(max_val) + 0.05
  )

print(wilcox_results_main_taxa)

# Select only the columns you want in the CSV
wilcox_csv_taxa <- wilcox_results_main_taxa %>%
  select(ML_Number, Metric, group1, group2, p_value, p.adj, p.adj.signif)

# Export to CSV
write.csv(
  wilcox_csv_taxa,
  "wilcoxon_results_main_taxa.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)

# descriptive stats
descriptive_stats_main_taxa <- long_data_all_main_taxa %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n        = sum(!is.na(Value)),
    mean     = mean(Value, na.rm = TRUE),
    median   = median(Value, na.rm = TRUE),
    sd       = sd(Value, na.rm = TRUE),
    IQR      = IQR(Value, na.rm = TRUE),
    min      = min(Value, na.rm = TRUE),
    max      = max(Value, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  descriptive_stats_main_taxa,
  "descriptive_stats_taxa_all_main.csv",
  row.names = FALSE
)

print(descriptive_stats_main_taxa)


# mixed modelling
auc_taxa <- lmer(
  Value ~ Rarefaction + (1 | ML_Number),
  data = long_data_taxa %>%
    filter(Metric == "AUC")
)

summary(auc_taxa)

ba_taxa <- lmer(
  Value ~ Rarefaction + (1 | ML_Number),
  data = long_data_taxa %>%
    filter(Metric == "Balanced_Accuracy")
)

summary(ba_taxa)


# ---- Plot ----
p_main_taxa <- ggplot(long_data_all_main_taxa,
                aes(x = Rarefaction,
                    y = Value,
                    fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ ML_Number) +
  stat_pvalue_manual(wilcox_results_main_taxa,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Taxa: Rarefied vs Non-rarefied (Paired Wilcoxon)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
# ---- Save ----
ggsave("taxa_rare_nrare_metrics.pdf",
       p_main_taxa,
       width = 14,
       height = 10)

print(p_main_taxa)

# supplementary metrics
# Shapiro-Wilk per ML_Number × Metric × Rarefaction for normality
normality_results_supp_taxa <- long_data_all_supp_taxa %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) {  # check for zero variance
        NA
      } else {
        shapiro.test(Value)$p.value
      }
    } else {
      NA
    },
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
      shapiro_p > 0.05 ~ "Normal",
      shapiro_p <= 0.05 ~ "Not normal"
    )
  )

print(normality_results_supp_taxa)

# Select only the columns you want in the CSV
normality_csv_taxa <- normality_results_supp_taxa %>%
  select(ML_Number, Metric, Rarefaction, n, shapiro_p, normality)

# Export to CSV
write.csv(
  normality_csv_taxa,
  "normality_results_supp_taxa.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)


# -------------------------
# Run paired Wilcoxon test for taxa
# -------------------------
wilcox_results_supp_taxa <- long_data_all_supp_taxa %>%
  group_by(ML_Number, Metric) %>%
  summarise(
    p_value = wilcox.test(
      Value[Rarefaction == "Rarefied"],
      Value[Rarefaction == "Non-rarefied"],
      paired = FALSE
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p.adj = p.adjust(p_value, method = "BH"),
    p.adj.signif = case_when(
      p.adj <= 0.001 ~ "***",
      p.adj <= 0.01  ~ "**",
      p.adj <= 0.05  ~ "*",
      TRUE           ~ "ns"
    ),
    group1 = "Non-rarefied",  # first box
    group2 = "Rarefied",      # second box
    # y.position slightly above max of this ML_Number × Metric
    y.position = long_data_all_supp_taxa %>%
      filter(ML_Number == unique(ML_Number),
             Metric == unique(Metric)) %>%
      summarise(max_val = max(Value)) %>%
      pull(max_val) + 0.05
  )

print(wilcox_results_supp_taxa)

# Select only the columns you want in the CSV
wilcox_csv_taxa <- wilcox_results_supp_taxa %>%
  select(ML_Number, Metric, group1, group2, p_value, p.adj, p.adj.signif)

# Export to CSV
write.csv(
  wilcox_csv_taxa,
  "wilcoxon_results_supp_taxa.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)

# descriptive stats
descriptive_stats_supp_taxa <- long_data_all_supp_taxa %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n        = sum(!is.na(Value)),
    mean     = mean(Value, na.rm = TRUE),
    median   = median(Value, na.rm = TRUE),
    sd       = sd(Value, na.rm = TRUE),
    IQR      = IQR(Value, na.rm = TRUE),
    min      = min(Value, na.rm = TRUE),
    max      = max(Value, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  descriptive_stats_supp_taxa,
  "descriptive_stats_taxa_all_supp.csv",
  row.names = FALSE
)

print(descriptive_stats_supp_taxa)



# ---- Plot ----
p_supp_taxa <- ggplot(long_data_all_supp_taxa,
                aes(x = Rarefaction,
                    y = Value,
                    fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ ML_Number) +
  stat_pvalue_manual(wilcox_results_supp_taxa,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Taxa: Rarefied vs Non-rarefied (Paired Wilcoxon)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
# ---- Save ----
ggsave("taxa_rare_nrare_metrics.pdf",
       p_asv,
       width = 14,
       height = 10)

print(p_taxa)






# Shapiro-Wilk per ML_Number × Metric × Rarefaction for normality
# normality_results_taxa <- long_data_taxa %>%
#   group_by(ML_Number, Metric, Rarefaction) %>%
#   summarise(
#     n = n(),
#     shapiro_p = if (n >= 3 & n <= 5000) {
#       if (var(Value) == 0) {  # check for zero variance
#         NA
#       } else {
#         shapiro.test(Value)$p.value
#       }
#     } else {
#       NA
#     },
#     .groups = "drop"
#   ) %>%
#   mutate(
#     normality = case_when(
#       is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
#       shapiro_p > 0.05 ~ "Normal",
#       shapiro_p <= 0.05 ~ "Not normal"
#     )
#   )
# 
# print(normality_results_taxa)
# 
# # Select only the columns you want in the CSV
# normality_csv_taxa <- normality_results_taxa %>%
#   select(ML_Number, Metric, Rarefaction, n, shapiro_p, normality)
# 
# # Export to CSV
# write.csv(
#   normality_csv_taxa,
#   "normality_results_taxa.csv",  # output file name
#   row.names = FALSE        # do not include row numbers
# )
# 
# 
# # -------------------------
# # Run paired Wilcoxon test for taxa
# # -------------------------
# wilcox_results_taxa <- long_data_taxa %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = TRUE
#     )$p.value,
#     .groups = "drop"
#   ) %>%
#   mutate(
#     p.adj = p.adjust(p_value, method = "BH"),
#     p.adj.signif = case_when(
#       p.adj <= 0.001 ~ "***",
#       p.adj <= 0.01  ~ "**",
#       p.adj <= 0.05  ~ "*",
#       TRUE           ~ "ns"
#     ),
#     group1 = "Non-rarefied",  # first box
#     group2 = "Rarefied",      # second box
#     # y.position slightly above max of this ML_Number × Metric
#     y.position = long_data_taxa %>%
#       filter(ML_Number == unique(ML_Number),
#              Metric == unique(Metric)) %>%
#       summarise(max_val = max(Value)) %>%
#       pull(max_val) + 0.05
#   )
# 
# print(wilcox_results_taxa)
# 
# # Select only the columns you want in the CSV
# wilcox_csv_taxa <- wilcox_results_taxa %>%
#   select(ML_Number, Metric, group1, group2, p_value, p.adj, p.adj.signif)
# 
# # Export to CSV
# write.csv(
#   wilcox_csv_taxa,
#   "wilcoxon_results_taxa.csv",  # output file name
#   row.names = FALSE        # do not include row numbers
# )
# 
# # ---- Plot ----
# p_taxa <- ggplot(long_data_taxa,
#                 aes(x = Rarefaction,
#                     y = Value,
#                     fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_results,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Taxa: Rarefied vs Non-rarefied (Paired Wilcoxon)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# # ---- Save ----
# ggsave("taxa_rare_nrare_metrics.pdf",
#        p_taxa,
#        width = 14,
#        height = 10)
# 
# print(p_taxa)
# 
# # ----------------------- Normality test across all papers ----------------------
# # Shapiro-Wilk per metric, all papers combined
# normality_combined_taxa <- long_data_taxa %>%
#   group_by(Metric, Rarefaction) %>%
#   summarise(
#     n = n(),
#     shapiro_p = if (n >= 3 & n <= 5000) {
#       if (var(Value) == 0) NA else shapiro.test(Value)$p.value
#     } else NA,
#     .groups = "drop"
#   ) %>%
#   mutate(
#     normality = case_when(
#       is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
#       shapiro_p > 0.05 ~ "Normal",
#       TRUE ~ "Not normal"
#     )
#   )
# 
# # Export normality table
# write.csv(
#   normality_combined_taxa,
#   "normality_results_taxa_combined.csv",
#   row.names = FALSE
# )
# 
# # ----------------------- Wilcoxon test across all papers -----------------------
# wilcox_combined_taxa <- long_data_taxa %>%
#   group_by(Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = FALSE  # not paired across papers
#     )$p.value,
#     n_nonrarefied = sum(Rarefaction == "Non-rarefied"),
#     n_rarefied = sum(Rarefaction == "Rarefied"),
#     .groups = "drop"
#   ) %>%
#   mutate(
#     p.adj = p.adjust(p_value, method = "BH"),
#     p.adj.signif = case_when(
#       p.adj <= 0.001 ~ "***",
#       p.adj <= 0.01  ~ "**",
#       p.adj <= 0.05  ~ "*",
#       TRUE ~ "ns"
#     ),
#     group1 = "Non-rarefied",
#     group2 = "Rarefied",
#     # y.position slightly above max of all values per metric
#     y.position = long_data_taxa %>%
#       group_by(Metric) %>%
#       summarise(max_val = max(Value), .groups = "drop") %>%
#       pull(max_val) + 0.05
#   )
# 
# # Export Wilcoxon table
# wilcox_csv_combined <- wilcox_combined_taxa %>%
#   select(Metric, group1, group2, n_nonrarefied, n_rarefied, p_value, p.adj, p.adj.signif)
# 
# write.csv(
#   wilcox_csv_combined,
#   "wilcoxon_results_taxa_combined.csv",
#   row.names = FALSE
# )
# 
# # ----------------------- Plot -----------------------
# p_taxa_combined <- ggplot(long_data_taxa,
#                           aes(x = Rarefaction,
#                               y = Value,
#                               fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_wrap(~ Metric, scales = "free_y") +
#   stat_pvalue_manual(wilcox_combined_taxa,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Taxa: Rarefied vs Non-rarefied (All Papers Combined)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# # ----------------------- Save plot -----------------------
# ggsave(
#   filename = "taxa_rare_nrare_metrics_combined.tiff",  # TIFF output
#   plot = p_taxa_combined,
#   width = 10,       # width in inches
#   height = 6,       # height in inches
#   dpi = 300,        # high resolution
#   compression = "lzw"  # optional, lossless compression
# )
# 
# print(p_taxa_combined)
# 
# # Create combined dataset
# combined_data_taxa <- long_data_taxa %>%
#   mutate(ML_Number = "All")   # Add new pseudo-paper
# 
# long_data_taxa_extended <- bind_rows(
#   long_data_taxa,
#   combined_data_taxa
# )
# 
# long_data_taxa_extended$ML_Number <- as.character(long_data_taxa_extended$ML_Number)
# 
# long_data_taxa_extended$ML_Number <- factor(
#   long_data_taxa_extended$ML_Number,
#   levels = c(as.character(1:16), "All")
# )
# 
# long_data_taxa_extended <- long_data_taxa_extended %>%
#   filter(!is.na(ML_Number))
# 
# # Per paper (paired)
# wilcox_papers <- long_data_taxa_extended %>%
#   filter(ML_Number != "All") %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = TRUE
#     )$p.value,
#     .groups = "drop"
#   )
# 
# # Combined (unpaired)
# wilcox_all <- long_data_taxa_extended %>%
#   filter(ML_Number == "All") %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = FALSE
#     )$p.value,
#     .groups = "drop"
#   )
# 
# wilcox_results_taxa_extended <- bind_rows(wilcox_papers, wilcox_all) %>%
#   mutate(
#     p.adj = p.adjust(p_value, method = "BH"),
#     p.adj.signif = case_when(
#       p.adj <= 0.001 ~ "***",
#       p.adj <= 0.01  ~ "**",
#       p.adj <= 0.05  ~ "*",
#       TRUE ~ "ns"
#     ),
#     group1 = "Non-rarefied",
#     group2 = "Rarefied"
#   )
# 
# # Compute max y per panel
# y_positions <- long_data_taxa_extended %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     y.position = max(Value, na.rm = TRUE) + 0.05,
#     .groups = "drop"
#   )
# 
# # Join onto Wilcoxon table
# wilcox_results_taxa_extended <- wilcox_results_taxa_extended %>%
#   left_join(y_positions, by = c("ML_Number", "Metric"))
# 
# p_taxa_full <- ggplot(long_data_taxa_extended,
#                       aes(x = Rarefaction,
#                           y = Value,
#                           fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_results_taxa_extended,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Taxa: Rarefied vs Non-rarefied (Individual Papers + Combined)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# 
# 
# long_data_main <- long_data_taxa_extended %>%
#   filter(Metric %in% main_metrics)
# 
# wilcox_main <- wilcox_results_taxa_extended %>%
#   filter(Metric %in% main_metrics)
# 
# p_taxa_main <- ggplot(long_data_main,
#                       aes(x = Rarefaction,
#                           y = Value,
#                           fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_main,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Taxa: Rarefied vs Non-rarefied (AUC & Balanced Accuracy)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# 
# long_data_supp <- long_data_taxa_extended %>%
#   filter(Metric %in% supplementary_metrics)
# 
# wilcox_supp <- wilcox_results_taxa_extended %>%
#   filter(Metric %in% supplementary_metrics)
# 
# p_taxa_supp <- ggplot(long_data_supp,
#                       aes(x = Rarefaction,
#                           y = Value,
#                           fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_supp,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Taxa: Rarefied vs Non-rarefied (Sensitivity & Specificity)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# print(p_taxa_supp)
# print(p_taxa_main)

# -------------------------- asv --------------------------
long_data_asv <- long_data_no_outliers %>%
  dplyr::filter(grepl("asv$", Dataset))

# Create combined dataset
combined_data_asv <- long_data_asv %>%
  mutate(ML_Number = "All")   # Add new pseudo-paper

long_data_asv_extended <- bind_rows(
  long_data_asv,
  combined_data_asv
)

long_data_asv_extended$ML_Number <- as.character(long_data_asv_extended$ML_Number)

long_data_asv_extended$ML_Number <- factor(
  long_data_asv_extended$ML_Number,
  levels = c(as.character(1:16), "All")
)

long_data_asv_extended <- long_data_asv_extended %>%
  filter(!is.na(ML_Number))

# Separate into main and supplementary metrics 
long_data_all_main_asv <- long_data_asv_extended %>%
  filter(Metric %in% main_metrics)

long_data_all_supp_asv <- long_data_asv_extended %>%
  filter(Metric %in% supplementary_metrics)

# main metrics
# Shapiro-Wilk per ML_Number × Metric × Rarefaction for normality
normality_results_main_asv <- long_data_all_main_asv %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) {  # check for zero variance
        NA
      } else {
        shapiro.test(Value)$p.value
      }
    } else {
      NA
    },
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
      shapiro_p > 0.05 ~ "Normal",
      shapiro_p <= 0.05 ~ "Not normal"
    )
  )

print(normality_results_main_asv)

# Select only the columns you want in the CSV
normality_csv_asv <- normality_results_main_asv %>%
  select(ML_Number, Metric, Rarefaction, n, shapiro_p, normality)

# Export to CSV
write.csv(
  normality_csv_asv,
  "normality_results_asv.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)


# -------------------------
# Run paired Wilcoxon test for taxa
# -------------------------
wilcox_results_main_asv <- long_data_all_main_asv %>%
  group_by(ML_Number, Metric) %>%
  summarise(
    p_value = wilcox.test(
      Value[Rarefaction == "Rarefied"],
      Value[Rarefaction == "Non-rarefied"],
      paired = TRUE
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p.adj = p.adjust(p_value, method = "BH"),
    p.adj.signif = case_when(
      p.adj <= 0.001 ~ "***",
      p.adj <= 0.01  ~ "**",
      p.adj <= 0.05  ~ "*",
      TRUE           ~ "ns"
    ),
    group1 = "Non-rarefied",  # first box
    group2 = "Rarefied",      # second box
    # y.position slightly above max of this ML_Number × Metric
    y.position = long_data_all_main_asv %>%
      filter(ML_Number == unique(ML_Number),
             Metric == unique(Metric)) %>%
      summarise(max_val = max(Value)) %>%
      pull(max_val) + 0.05
  )

print(wilcox_results_main_asv)

# Select only the columns you want in the CSV
wilcox_csv_asv <- wilcox_results_main_asv %>%
  select(ML_Number, Metric, group1, group2, p_value, p.adj, p.adj.signif)

# Export to CSV
write.csv(
  wilcox_csv_asv,
  "wilcoxon_results_main_asv.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)

# descriptive stats
descriptive_stats_main_asv <- long_data_all_main_asv %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n        = sum(!is.na(Value)),
    mean     = mean(Value, na.rm = TRUE),
    median   = median(Value, na.rm = TRUE),
    sd       = sd(Value, na.rm = TRUE),
    IQR      = IQR(Value, na.rm = TRUE),
    min      = min(Value, na.rm = TRUE),
    max      = max(Value, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  descriptive_stats_main_asv,
  "descriptive_stats_asv_all_main.csv",
  row.names = FALSE
)

print(descriptive_stats_main_asv)

auc_main_asv <- lmer(
  Value ~ Rarefaction + (1 | ML_Number),
  data = long_data_asv %>%
    filter(Metric == "AUC")
)

summary(auc_main_asv)

ba_main_asv <- lmer(
  Value ~ Rarefaction + (1 | ML_Number),
  data = long_data_asv %>%
    filter(Metric == "Balanced_Accuracy")
)

summary(ba_main_asv)

# ---- Plot ----
p_main_asv <- ggplot(long_data_all_main_asv,
                 aes(x = Rarefaction,
                     y = Value,
                     fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ ML_Number) +
  stat_pvalue_manual(wilcox_results_main_asv,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Asv: Rarefied vs Non-rarefied (Paired Wilcoxon)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
# ---- Save ----
ggsave("asv_rare_nrare_metrics.pdf",
       p_asv,
       width = 14,
       height = 10)

print(p_asv)

# supplementary metrics
# Shapiro-Wilk per ML_Number × Metric × Rarefaction for normality
normality_results_supp_asv <- long_data_all_supp_asv %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) {  # check for zero variance
        NA
      } else {
        shapiro.test(Value)$p.value
      }
    } else {
      NA
    },
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
      shapiro_p > 0.05 ~ "Normal",
      shapiro_p <= 0.05 ~ "Not normal"
    )
  )

print(normality_results_supp_asv)

# Select only the columns you want in the CSV
normality_csv_asv <- normality_results_supp_asv %>%
  select(ML_Number, Metric, Rarefaction, n, shapiro_p, normality)

# Export to CSV
write.csv(
  normality_csv_asv,
  "normality_results_supp_asv.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)


# -------------------------
# Run paired Wilcoxon test for taxa
# -------------------------
wilcox_results_supp_asv <- long_data_all_supp_asv %>%
  group_by(ML_Number, Metric) %>%
  summarise(
    p_value = wilcox.test(
      Value[Rarefaction == "Rarefied"],
      Value[Rarefaction == "Non-rarefied"],
      paired = FALSE
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p.adj = p.adjust(p_value, method = "BH"),
    p.adj.signif = case_when(
      p.adj <= 0.001 ~ "***",
      p.adj <= 0.01  ~ "**",
      p.adj <= 0.05  ~ "*",
      TRUE           ~ "ns"
    ),
    group1 = "Non-rarefied",  # first box
    group2 = "Rarefied",      # second box
    # y.position slightly above max of this ML_Number × Metric
    y.position = long_data_all_supp_asv %>%
      filter(ML_Number == unique(ML_Number),
             Metric == unique(Metric)) %>%
      summarise(max_val = max(Value)) %>%
      pull(max_val) + 0.05
  )

print(wilcox_results_supp_asv)

# Select only the columns you want in the CSV
wilcox_csv_asv <- wilcox_results_supp_asv %>%
  select(ML_Number, Metric, group1, group2, p_value, p.adj, p.adj.signif)

# Export to CSV
write.csv(
  wilcox_csv_asv,
  "wilcoxon_results_supp_asv.csv",  # output file name
  row.names = FALSE        # do not include row numbers
)

descriptive_stats_supp_asv <- long_data_all_supp_asv %>%
  group_by(ML_Number, Metric, Rarefaction) %>%
  summarise(
    n        = sum(!is.na(Value)),
    mean     = mean(Value, na.rm = TRUE),
    median   = median(Value, na.rm = TRUE),
    sd       = sd(Value, na.rm = TRUE),
    IQR      = IQR(Value, na.rm = TRUE),
    min      = min(Value, na.rm = TRUE),
    max      = max(Value, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  descriptive_stats_supp_asv,
  "descriptive_stats_asv_all_supp.csv",
  row.names = FALSE
)

print(descriptive_stats_supp_asv)

# ---- Plot ----
p_asv <- ggplot(long_data_all_supp_asv,
                aes(x = Rarefaction,
                    y = Value,
                    fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ ML_Number) +
  stat_pvalue_manual(wilcox_results_supp_asv,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Asv: Rarefied vs Non-rarefied (Paired Wilcoxon)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
# ---- Save ----
ggsave("asv_rare_nrare_metrics.pdf",
       p_asv,
       width = 14,
       height = 10)

print(p_asv)

# ----------------------- Normality test across all papers ----------------------
# Shapiro-Wilk per metric, all papers combined
# normality_combined_asv <- long_data_asv %>%
#   group_by(Metric, Rarefaction) %>%
#   summarise(
#     n = n(),
#     shapiro_p = if (n >= 3 & n <= 5000) {
#       if (var(Value) == 0) NA else shapiro.test(Value)$p.value
#     } else NA,
#     .groups = "drop"
#   ) %>%
#   mutate(
#     normality = case_when(
#       is.na(shapiro_p) ~ "Not tested (n<3, n>5000, or zero variance)",
#       shapiro_p > 0.05 ~ "Normal",
#       TRUE ~ "Not normal"
#     )
#   )
# 
# # Export normality table
# write.csv(
#   normality_combined_asv,
#   "normality_results_asv_combined.csv",
#   row.names = FALSE
# )
# 
# # ----------------------- Wilcoxon test across all papers -----------------------
# wilcox_combined_asv <- long_data_asv %>%
#   group_by(Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = FALSE  # not paired across papers
#     )$p.value,
#     n_nonrarefied = sum(Rarefaction == "Non-rarefied"),
#     n_rarefied = sum(Rarefaction == "Rarefied"),
#     .groups = "drop"
#   ) %>%
#   mutate(
#     p.adj = p.adjust(p_value, method = "BH"),
#     p.adj.signif = case_when(
#       p.adj <= 0.001 ~ "***",
#       p.adj <= 0.01  ~ "**",
#       p.adj <= 0.05  ~ "*",
#       TRUE ~ "ns"
#     ),
#     group1 = "Non-rarefied",
#     group2 = "Rarefied",
#     # y.position slightly above max of all values per metric
#     y.position = long_data_asv %>%
#       group_by(Metric) %>%
#       summarise(max_val = max(Value), .groups = "drop") %>%
#       pull(max_val) + 0.05
#   )
# 
# # Export Wilcoxon table
# wilcox_csv_combined <- wilcox_combined_asv %>%
#   select(Metric, group1, group2, n_nonrarefied, n_rarefied, p_value, p.adj, p.adj.signif)
# 
# write.csv(
#   wilcox_csv_combined,
#   "wilcoxon_results_asv_combined.csv",
#   row.names = FALSE
# )
# 
# # ----------------------- Plot -----------------------
# p_asv_combined <- ggplot(long_data_asv,
#                           aes(x = Rarefaction,
#                               y = Value,
#                               fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_wrap(~ Metric, scales = "free_y") +
#   stat_pvalue_manual(wilcox_combined_asv,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Asv: Rarefied vs Non-rarefied (All Papers Combined)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# # ----------------------- Save plot -----------------------
# ggsave(
#   filename = "asv_rare_nrare_metrics_combined.tiff",  # TIFF output
#   plot = p_asv_combined,
#   width = 10,       # width in inches
#   height = 6,       # height in inches
#   dpi = 300,        # high resolution
#   compression = "lzw"  # optional, lossless compression
# )
# 
# print(p_asv_combined)
# 
# # Create combined dataset
# combined_data <- long_data_asv %>%
#   mutate(ML_Number = "All")   # Add new pseudo-paper
# 
# long_data_asv_extended <- bind_rows(
#   long_data_asv,
#   combined_data
# )
# 
# long_data_asv_extended$ML_Number <- as.character(long_data_asv_extended$ML_Number)
# 
# long_data_asv_extended$ML_Number <- factor(
#   long_data_asv_extended$ML_Number,
#   levels = c(as.character(1:16), "All")
# )
# 
# long_data_asv_extended <- long_data_asv_extended %>%
#   filter(!is.na(ML_Number))
# 
# # Per paper (paired)
# wilcox_papersasv <- long_data_asv_extended %>%
#   filter(ML_Number != "All") %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = TRUE
#     )$p.value,
#     .groups = "drop"
#   )
# 
# # Combined (unpaired)
# wilcox_allasv <- long_data_asv_extended %>%
#   filter(ML_Number == "All") %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     p_value = wilcox.test(
#       Value[Rarefaction == "Rarefied"],
#       Value[Rarefaction == "Non-rarefied"],
#       paired = FALSE
#     )$p.value,
#     .groups = "drop"
#   )
# 
# wilcox_results_asv_extended <- bind_rows(wilcox_papers, wilcox_allasv) %>%
#   mutate(
#     p.adj = p.adjust(p_value, method = "BH"),
#     p.adj.signif = case_when(
#       p.adj <= 0.001 ~ "***",
#       p.adj <= 0.01  ~ "**",
#       p.adj <= 0.05  ~ "*",
#       TRUE ~ "ns"
#     ),
#     group1 = "Non-rarefied",
#     group2 = "Rarefied"
#   )
# 
# # Compute max y per panel
# y_positions_asv <- long_data_asv_extended %>%
#   group_by(ML_Number, Metric) %>%
#   summarise(
#     y.position = max(Value, na.rm = TRUE) + 0.05,
#     .groups = "drop"
#   )
# 
# # Join onto Wilcoxon table
# wilcox_results_asv_extended <- wilcox_results_asv_extended %>%
#   left_join(y_positions, by = c("ML_Number", "Metric"))
# 
# p_asv_full <- ggplot(long_data_asv_extended,
#                       aes(x = Rarefaction,
#                           y = Value,
#                           fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_results_asv_extended,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Asv: Rarefied vs Non-rarefied (Individual Papers + Combined)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# ggsave(
#   filename = "asv_metrics_combined.tiff",  # TIFF output
#   plot = p_asv_full,
#   width = 10,       # width in inches
#   height = 6,       # height in inches
#   dpi = 300,        # high resolution
#   compression = "lzw"  # optional, lossless compression
# )
# 
# 
# long_data_main_asv <- long_data_asv_extended %>%
#   filter(Metric %in% main_metrics)
# 
# wilcox_main_asv <- wilcox_results_asv_extended %>%
#   filter(Metric %in% main_metrics)
# 
# p_asv_main <- ggplot(long_data_main_asv,
#                       aes(x = Rarefaction,
#                           y = Value,
#                           fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_main_asv,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Asv: Rarefied vs Non-rarefied (AUC & Balanced Accuracy)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# ggsave(
#   filename = "auc_balacc_asv_all.tiff",  # TIFF output
#   plot = p_asv_main,
#   width = 10,       # width in inches
#   height = 6,       # height in inches
#   dpi = 300,        # high resolution
#   compression = "lzw"  # optional, lossless compression
# )
# 
# 
# 
# long_data_supp_asv <- long_data_asv_extended %>%
#   filter(Metric %in% supplementary_metrics)
# 
# wilcox_supp_asv <- wilcox_results_asv_extended %>%
#   filter(Metric %in% supplementary_metrics)
# 
# p_asv_supp <- ggplot(long_data_supp_asv,
#                       aes(x = Rarefaction,
#                           y = Value,
#                           fill = Rarefaction)) +
#   geom_boxplot(alpha = 0.7, outlier.shape = NA) +
#   geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
#   facet_grid(Metric ~ ML_Number) +
#   stat_pvalue_manual(wilcox_supp_asv,
#                      label = "p.adj.signif",
#                      tip.length = 0.01) +
#   theme_bw(base_size = 12) +
#   labs(title = "Asv: Rarefied vs Non-rarefied (Sensitivity & Specificity)",
#        y = "Performance",
#        x = "") +
#   scale_fill_manual(values = c("#D55E00", "#0072B2")) +
#   theme(
#     legend.position = "none",
#     strip.background = element_rect(fill = "grey90"),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )
# 
# ggsave(
#   filename = "sensitivity_specificity_asv_all.tiff",  # TIFF output
#   plot = p_asv_supp,
#   width = 10,       # width in inches
#   height = 6,       # height in inches
#   dpi = 300,        # high resolution
#   compression = "lzw"  # optional, lossless compression
# )
# 
# print(p_asv_supp)
# print(p_asv_main)


