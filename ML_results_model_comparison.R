# Install if needed
# install.packages(c("tidyverse", "readr", "viridis", "patchwork", "dplyr", "tidyr", "lme4", "lmerTest", "ggpubr", "rstatix"))

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
# This has been written to compare the effect of rarefaction on 4 different machine learning models using taxa and asv data
# The analysis can be performed with and without outliers (filtering based on IQR for skewed data or z-score for normal data)

# ===============================
# Model specific Comparison Boxplots
# ===============================
# ---- Load data ----
data <- read.csv("ml_summary_means.csv", stringsAsFactors = FALSE)

data <- data %>%
  mutate(
    Rarefaction = ifelse(grepl("nonrarefied", Dataset),
                         "Non-rarefied",
                         "Rarefied"),
    Feature_Type = ifelse(grepl("asv", Dataset),
                          "ASV",
                          "Taxa"),
    Model = factor(Model)
  )


metrics <- c("AUC", "Sensitivity", "Specificity", "Balanced_Accuracy")
main_metrics <- c("AUC", "Balanced_Accuracy")
supplementary_metrics <- c("Sensitivity", "Specificity")

# WITH OUTLIERS #
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

# --------- Taxa ----------
# WILCOX TEST WILL NEED TO BE CHANGED FROM PAIRED TO UNPAIRED
long_data_taxa <- long_data_no_outliers %>% # CHANGE TO INCLUDE OR REMOVE OUTLIERS
  filter(Feature_Type == "Taxa")

long_data_main_taxa <- long_data_taxa %>%
  filter(Metric %in% main_metrics)


long_data_supp_taxa <- long_data_taxa %>%
  filter(Metric %in% supplementary_metrics)

normality_results_main_taxa_model <- long_data_main_taxa %>%
  group_by(Model, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) NA else shapiro.test(Value)$p.value
    } else NA,
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested",
      shapiro_p > 0.05 ~ "Normal",
      TRUE ~ "Not normal"
    )
  )

write.csv(normality_results_main_taxa_model,
          "normality_results_taxa_per_model.csv",
          row.names = FALSE)

wilcox_results_main_taxa_model <- long_data_main_taxa %>%
  group_by(Model, Metric) %>%
  summarise(
    p_value = wilcox.test(
      Value[Rarefaction == "Rarefied"],
      Value[Rarefaction == "Non-rarefied"],
      paired = FALSE # CHANGE IF YOU ARE REMOVING OUTLIERS TO MAKE IT UNPAIRED
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p.adj = p.adjust(p_value, method = "BH"),
    p.adj.signif = case_when(
      p.adj <= 0.001 ~ "***",
      p.adj <= 0.01  ~ "**",
      p.adj <= 0.05  ~ "*",
      TRUE ~ "ns"
    ),
    group1 = "Non-rarefied",
    group2 = "Rarefied"
  )

write.csv(wilcox_results_main_taxa_model,
          "wilcox_results_main_taxa_per_model.csv",
          row.names = FALSE)

y_positions_model <- long_data_main_taxa %>%
  group_by(Model, Metric) %>%
  summarise(
    y.position = max(Value, na.rm = TRUE) + 0.05,
    .groups = "drop"
  )

wilcox_results_main_taxa_model <- wilcox_results_main_taxa_model %>%
  left_join(y_positions_model, by = c("Model", "Metric"))

p_taxa_main_model <- ggplot(long_data_main_taxa,
                       aes(x = Rarefaction,
                           y = Value,
                           fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ Model) +
  stat_pvalue_manual(wilcox_results_main_taxa_model,
                     y.position = "y.position",
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Taxa: Rarefied vs Non-rarefied (Per Model Type)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90")
  )

ggsave("taxa_per_main_model.tiff",
       p_taxa_model,
       width = 14,
       height = 8,
       dpi = 300,
       compression = "lzw")

## Supplementary
# WILCOX TEST WILL NEED TO BE CHANGED FROM PAIRED TO UNPAIRED
long_data_supp_taxa <- long_data_taxa %>%
  filter(Metric %in% supplementary_metrics)


normality_results_supp_taxa_model <- long_data_supp_taxa %>%
  group_by(Model, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) NA else shapiro.test(Value)$p.value
    } else NA,
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested",
      shapiro_p > 0.05 ~ "Normal",
      TRUE ~ "Not normal"
    )
  )

write.csv(normality_results_supp_taxa_model,
          "normality_results_taxa_supp_per_model.csv",
          row.names = FALSE)

wilcox_results_supp_taxa_model <- long_data_supp_taxa %>%
  group_by(Model, Metric) %>%
  summarise(
    p_value = wilcox.test(
      Value[Rarefaction == "Rarefied"],
      Value[Rarefaction == "Non-rarefied"],
      paired = FALSE # CHANGE IF YOU ARE REMOVING OUTLIERS TO MAKE IT UNPAIRED
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p.adj = p.adjust(p_value, method = "BH"),
    p.adj.signif = case_when(
      p.adj <= 0.001 ~ "***",
      p.adj <= 0.01  ~ "**",
      p.adj <= 0.05  ~ "*",
      TRUE ~ "ns"
    ),
    group1 = "Non-rarefied",
    group2 = "Rarefied"
  )

write.csv(wilcox_results_supp_taxa_model,
          "wilcox_results_supp_taxa_per_model.csv",
          row.names = FALSE)

y_positions_model <- long_data_supp_taxa %>%
  group_by(Model, Metric) %>%
  summarise(
    y.position = max(Value, na.rm = TRUE) + 0.05,
    .groups = "drop"
  )

wilcox_results_supp_taxa_model <- wilcox_results_supp_taxa_model %>%
  left_join(y_positions_model, by = c("Model", "Metric"))

p_taxa_supp_model <- ggplot(long_data_supp_taxa,
                            aes(x = Rarefaction,
                                y = Value,
                                fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ Model) +
  stat_pvalue_manual(wilcox_results_supp_taxa_model,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Taxa: Rarefied vs Non-rarefied (Per Model Type)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90")
  )

ggsave("taxa_supp_model.tiff",
       p_taxa_model,
       width = 14,
       height = 8,
       dpi = 300,
       compression = "lzw")

# --------- Asv ----------
long_data_asv <- long_data_no_outliers %>% # CHANGE TO INCLUDE OR REMOVE OUTLIERS
  filter(Feature_Type == "ASV")

long_data_main_asv <- long_data_asv %>%
  filter(Metric %in% main_metrics)


long_data_supp_asv <- long_data_asv %>%
  filter(Metric %in% supplementary_metrics)


normality_results_main_asv_model <- long_data_main_asv %>%
  group_by(Model, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) NA else shapiro.test(Value)$p.value
    } else NA,
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested",
      shapiro_p > 0.05 ~ "Normal",
      TRUE ~ "Not normal"
    )
  )

write.csv(normality_results_main_asv_model,
          "normality_results_asv_per_model.csv",
          row.names = FALSE)

wilcox_results_main_asv_model <- long_data_main_asv %>%
  group_by(Model, Metric) %>%
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
      TRUE ~ "ns"
    ),
    group1 = "Non-rarefied",
    group2 = "Rarefied"
  )

write.csv(wilcox_results_main_asv_model,
          "wilcox_results_main_asv_per_model.csv",
          row.names = FALSE)

y_positions_model <- long_data_main_asv %>%
  group_by(Model, Metric) %>%
  summarise(
    y.position = max(Value, na.rm = TRUE) + 0.05,
    .groups = "drop"
  )

wilcox_results_main_asv_model <- wilcox_results_main_asv_model %>%
  left_join(y_positions_model, by = c("Model", "Metric"))

p_asv_main_model <- ggplot(long_data_main_asv,
                            aes(x = Rarefaction,
                                y = Value,
                                fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ Model) +
  stat_pvalue_manual(wilcox_results_main_asv_model,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Asv: Rarefied vs Non-rarefied (Per Model Type)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90")
  )

ggsave("asv_per_main_model.tiff",
       p_taxa_model,
       width = 14,
       height = 8,
       dpi = 300,
       compression = "lzw")

## Supplementary
long_data_supp_asv <- long_data_asv %>%
  filter(Metric %in% supplementary_metrics)


normality_results_supp_asv_model <- long_data_supp_asv %>%
  group_by(Model, Metric, Rarefaction) %>%
  summarise(
    n = n(),
    shapiro_p = if (n >= 3 & n <= 5000) {
      if (var(Value) == 0) NA else shapiro.test(Value)$p.value
    } else NA,
    .groups = "drop"
  ) %>%
  mutate(
    normality = case_when(
      is.na(shapiro_p) ~ "Not tested",
      shapiro_p > 0.05 ~ "Normal",
      TRUE ~ "Not normal"
    )
  )

write.csv(normality_results_supp_asv_model,
          "normality_results_asv_supp_per_model.csv",
          row.names = FALSE)

wilcox_results_supp_asv_model <- long_data_supp_asv %>%
  group_by(Model, Metric) %>%
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
      TRUE ~ "ns"
    ),
    group1 = "Non-rarefied",
    group2 = "Rarefied"
  )

write.csv(wilcox_results_supp_asv_model,
          "wilcox_results_supp_asv_per_model.csv",
          row.names = FALSE)

y_positions_model <- long_data_supp_asv %>%
  group_by(Model, Metric) %>%
  summarise(
    y.position = max(Value, na.rm = TRUE) + 0.05,
    .groups = "drop"
  )

wilcox_results_supp_asv_model <- wilcox_results_supp_asv_model %>%
  left_join(y_positions_model, by = c("Model", "Metric"))

p_asv_supp_model <- ggplot(long_data_supp_asv,
                            aes(x = Rarefaction,
                                y = Value,
                                fill = Rarefaction)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  facet_grid(Metric ~ Model) +
  stat_pvalue_manual(wilcox_results_supp_asv_model,
                     label = "p.adj.signif",
                     tip.length = 0.01) +
  theme_bw(base_size = 12) +
  labs(title = "Asv: Rarefied vs Non-rarefied (Per Model Type)",
       y = "Performance",
       x = "") +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90")
  )

ggsave("asv_supp_model.tiff",
       p_taxa_model,
       width = 14,
       height = 8,
       dpi = 300,
       compression = "lzw")
