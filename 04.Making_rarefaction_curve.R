# MY EDITS:
# REMOVED my_custom_theme, changed "type" and "site" variables to "sample_type" (matches my metadata)
library(gridExtra)
library(grid)


###rarefaction plot
rarefaction.df = data.frame(ASVs=rowSums(t(otu_table(physeq1))>0), reads=sample_sums(physeq1), sample_data(physeq1))
red_line <- 15000
blue_line <- red_line * 2

tiff("plot/102.reads.per.sample.tiff", units="in", width=4, height=3, res=600, compression = 'lzw')
ggplot(rarefaction.df, aes(x=reads)) + geom_histogram(bins=50, color='black', fill='grey') + 
    geom_vline(xintercept=red_line, color= "red", linetype='dashed') +
  geom_vline(xintercept=blue_line, color= "blue", linetype='dashed') +
  annotate("text", 
           x = red_line, y = Inf, 
           label = paste0(red_line," reads"), 
           hjust = -.1, vjust = 1.5,
           color = "red") +
  annotate("text", 
           x = blue_line, y = Inf, 
           label = paste0(blue_line," reads"), 
           hjust = -.1, vjust = 2.5,
           color = "blue") +
  coord_cartesian(clip = "off") +
  labs(title="Histogram: Reads per Sample") + xlab("Read Count") + ylab("Sample Count")
dev.off()

tiff("plot/103.reads.per.group.tiff", units="in", width=5, height=3, res=600, compression = 'lzw')
ggplot(rarefaction.df, aes(x = sample_type, y = reads, fill = sample_type, color=ASVs)) +
  geom_boxplot(color="black") + scale_fill_d3()+ scale_color_gradient( low = "blue", high = "red") +
  geom_point(position=position_jitterdodge(),alpha=0.5)+ xlab("")+ylab("Reads")+ 
  geom_hline(yintercept=red_line, color= "red", linetype='dashed') +
  geom_hline(yintercept=blue_line, color= "blue", linetype='dashed') +
  annotate("text", 
           x = Inf, y = red_line, 
           label = paste0(red_line," reads"), 
           hjust = 2.2, vjust = -0.5,
           color = "red") +
  annotate("text", 
           x = Inf, y = blue_line, 
           label = paste0(blue_line," reads"), 
           hjust = 2.2, vjust = -0.5,
           color = "blue") +
    ggtitle("Reads sequenced by SampleGroup")
dev.off()


# We use the rarefaction curve data produce by vegan above
out = vegan::rarecurve(as.data.frame(t(otu_table(physeq1))), step = red_line, label = F, 
                ylab="ASVs", lwd=.5, main="Rarefaction Curve for all samples")

names(out) = rownames(t(otu_table(physeq1)))

# Coerce data into "long" form.
protox <- mapply(FUN = function(x, y) {
  mydf <- as.data.frame(x)
  colnames(mydf) <- "value"
  mydf$SampleID <- y
  mydf$subsample <- attr(x, "Subsample")
  mydf
}, x = out, y = as.list(names(out)), SIMPLIFY = FALSE)

xy <- do.call(rbind, protox)
rownames(xy) <- NULL  # pretty
xy = data.frame(xy, sample_data(physeq1)[match(xy$SampleID, rownames(sample_data(physeq1))), ])


# Plot Rarefaction curve
tiff("plot/104.before_rarefaction.tiff", units="in", width=4, height=3, res=600, compression = 'lzw')
ggplot(xy, aes(x = subsample, y = value, color = SampleID)) +
  scale_color_discrete(guide = FALSE) +  # turn legend on or off
  geom_line() +
  geom_vline(xintercept=red_line, color= "red", linetype='dashed') + 
  geom_vline(xintercept=blue_line, color= "blue", linetype='dashed') + 
  labs(title="Rarefaction curves") + xlab("Sequenced Reads") + ylab('ASVs Detected')
  #facet_grid(type~site)
dev.off()


# Descriptive stats for exploring the samples not to be included after rarefaction
excluded_samples <- rarefaction.df %>% 
  filter(reads < red_line)

# Count total samples per sample_type
total_count <- rarefaction.df %>%
  count(sample_type) %>%
  rename(total_samples = n)

# Count excluded samples per sample_type
excluded_count <- excluded_samples %>%
  count(sample_type) %>%
  rename(excluded = n)

# Join starting from total_count to keep all classes
excluded_table <- total_count %>%
  left_join(excluded_count, by = "sample_type") %>%
  mutate(
    excluded = ifelse(is.na(excluded), 0, excluded),
    percent_excluded = (excluded / total_samples) * 100
  )

excluded_table

tiff("plot/107.excluded_samples_table.tiff", width = 6, height = 1, units = "in", res = 300)
grid.newpage()
if (nrow(excluded_table) == 0) {
  grid::grid.text(
    "No excluded samples",
    x = 0.5, y = 0.5,
    gp = grid::gpar(fontsize = 12)
  )
} else {
  tbl <- gridExtra::tableGrob(excluded_table, rows = NULL)
  grid::grid.draw(tbl)
}
grid::grid.text(
  paste0("Samples were rarefied to ", red_line," reads."),
  x = 0.5,
  y = 0.02,          # near bottom
  gp = grid::gpar(fontsize = 9)
)
dev.off()




###
readsPerSample = rowSums(t(otu_table(physeq1)))
fractionReadsAssigned = sapply(colnames(tax_table(physeq1)), function(x){
  rowSums(t(otu_table(physeq1))[, !is.na(tax_table(physeq1))[,x]]) / readsPerSample
})

fractionReadsAssigned = data.frame(SampleID = rownames(fractionReadsAssigned), fractionReadsAssigned)
fractionReadsAssigned.L = pivot_longer(fractionReadsAssigned, 
                                       cols=colnames(tax_table(physeq1)), names_to="taxlevel", values_to="fractionReadsAssigned")
fractionReadsAssigned.L = data.frame(fractionReadsAssigned.L, sample_data(physeq1)[fractionReadsAssigned.L$SampleID,])

fractionReadsAssigned.L$taxlevelf = factor(fractionReadsAssigned.L$taxlevel, levels=c("Kingdom","Phylum","Class","Order","Family","Genus","Species"))

# Boxplot, fraction assigned by SampleType
tiff("plot/105.read_assigned.tiff", units="in", width=6, height=3, res=600, compression = 'lzw')
ggplot(fractionReadsAssigned.L, aes(y = fractionReadsAssigned, x = taxlevelf))+
  geom_boxplot(color='black', outlier.shape=NA) +  
  scale_fill_d3()+ xlab("")+ylab("Fraction reads assigned")+
  ggtitle("Fraction of reads identified by taxonomic level")
dev.off()
