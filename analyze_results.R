library(ggplot2)
library(dplyr)
library(purrr)
library(readr)
library(rlang)
library(tidyverse)
library(knitr)
library(kableExtra)
library(xtable)
library(tidyr)  # Required for complete()


# Note to self: methods that are only for treatment effect of the treated entries: DID, DC-PR, MC-NNM, Mean Effect, Median Effect, KRLS, GANITE



# Function to read all CSV files in a folder
read_all_data <- function(folder_path) {
  file_list <- list.files(folder_path, full.names = TRUE, pattern = "\\.csv$")
  results <- do.call(rbind, lapply(file_list, read_csv))
  return(results)
}

# Read the results from the folder
results <- read_all_data("results_09_18")
results <- results %>%
  filter(!(grepl("^PaCE_", Method) & Method != "PaCE_40")) %>%  # Filter out all PaCE_X where X is not 40
  mutate(Method = ifelse(Method == "PaCE_40", "\\rowcolor{blue!15}PaCE", Method),
         Method = gsub("Causal Forest \\(R\\)", "Causal Forest", Method),
         Method = gsub("KRLS \\(R\\)", "KRLS", Method))  # Rename KRLS (R)


create_proportion_wins_table <- {
  
  # metric = "NMAE Treated Only"
  metric = "NMAE"
  
  calculate_best_proportions <- function(results) {
    all_methods <- unique(results$Method)  # Get all methods
    
    results %>%
      group_by(`Experiment ID`) %>%
      filter(!!sym(metric) == min(!!sym(metric))) %>%  # Use the provided metric for filtering
      ungroup() %>%
      count(Method) %>% # count number of wins
      complete(Method = all_methods, fill = list(n = 0)) %>%
      mutate(proportion = n / sum(n)) 
  }
  
  # Helper function to generate the results frame for each unit
  generate_unit_df <- function(unit_name) {
    unit_data <- results %>% filter(Unit == unit_name) %>%
      # Check if the global metric variable is "NMAE" and exclude specific methods
      filter(!(metric == "NMAE" & Method %in% c("DID", "DC-PR", "MC-NNM", "Mean Effect", "Median Effect", "KRLS", "GANITE")))
    
    
    # Calculate proportions for different conditions
    overall_best <- calculate_best_proportions(unit_data)
    empty_df = overall_best %>% mutate(empty="") %>% select(Method, empty)
    adaptive_no <- calculate_best_proportions(filter(unit_data, `Adaptive Treatment` == FALSE))
    adaptive_yes <- calculate_best_proportions(filter(unit_data, `Adaptive Treatment` == TRUE))
    frac_0_05 <- calculate_best_proportions(filter(unit_data, `Fraction Treated` == 0.05))
    frac_0_25 <- calculate_best_proportions(filter(unit_data, `Fraction Treated` == 0.25))
    frac_0_50 <- calculate_best_proportions(filter(unit_data, `Fraction Treated` == 0.5))
    frac_0_75 <- calculate_best_proportions(filter(unit_data, `Fraction Treated` == 0.75))
    frac_1_0 <- calculate_best_proportions(filter(unit_data, `Fraction Treated` == 1.0))
    effect_add <- calculate_best_proportions(filter(unit_data, `Treatment Effect Function` == "additive"))
    effect_mult <- calculate_best_proportions(filter(unit_data, `Treatment Effect Function` == "multiplicative"))
    
    # Select the top 5 methods based on overall best performance
    top_5_methods <- overall_best %>%
      arrange(desc(proportion)) %>%
      slice(1:6) %>%
      pull(Method)
    
    # Create the final results frame by joining all proportions based on methods
    final_table <- data.frame(Method = top_5_methods) %>%
      left_join(overall_best %>% select(Method, proportion) %>% rename(All = proportion), by = "Method") %>%
      left_join(empty_df %>% rename(empty1 = empty)) %>%
      left_join(adaptive_no %>% select(Method, proportion) %>% rename(Adaptive_No = proportion), by = "Method") %>%
      left_join(adaptive_yes %>% select(Method, proportion) %>% rename(Adaptive_Yes = proportion), by = "Method") %>%
      left_join(empty_df %>% rename(empty2 = empty)) %>%
      left_join(frac_0_05 %>% select(Method, proportion) %>% rename(`0.05` = proportion), by = "Method") %>%
      left_join(frac_0_25 %>% select(Method, proportion) %>% rename(`0.25` = proportion), by = "Method") %>%
      left_join(frac_0_50 %>% select(Method, proportion) %>% rename(`0.50` = proportion), by = "Method") %>%
      left_join(frac_0_75 %>% select(Method, proportion) %>% rename(`0.75` = proportion), by = "Method") %>%
      left_join(frac_1_0 %>% select(Method, proportion) %>% rename(`1.0` = proportion), by = "Method") %>%
      left_join(empty_df %>% rename(empty3 = empty)) %>%
      left_join(effect_add %>% select(Method, proportion) %>% rename(Additive = proportion), by = "Method") %>%
      left_join(effect_mult %>% select(Method, proportion) %>% rename(Multiplicative = proportion), by = "Method")
    final_table
  }
  
  # Generate the results for each unit
  snap_df <- generate_unit_df("CLIENT_ZIP")
  state_df <- generate_unit_df("State Name")
  county_df <- generate_unit_df("County Name")
  
  # Add the LaTeX formatted strings to the first entry of each block in the Method column
  snap_df$Method[1] <- paste0("\\addlinespace SNAP &&&&&&&\\\\ \\cline{1-1} \\addlinespace ", snap_df$Method[1])
  state_df$Method[1] <- paste0("\\addlinespace State &&&&&&&\\\\\ \\cline{1-1} \\addlinespace ", state_df$Method[1])
  county_df$Method[1] <- paste0("\\addlinespace County &&&&&&&\\\\ \\cline{1-1} \\addlinespace ", county_df$Method[1])
  
  # Combine the results frames for all units
  combined_df <- bind_rows(snap_df, state_df, county_df)
  combined_df[is.na(combined_df)] <- ""
  colnames(combined_df) <- c("", "All", "", "Y", "N", "", "0.05", "0.25", "0.50", "0.75", "1.0", "", "Add.", "Mult.")
  
  # Create the LaTeX table with escape = FALSE to allow LaTeX commands to pass through
  combined_results_latex <- kable(
    combined_df, "latex", booktabs = TRUE, 
    caption = "Proportion of instances in which each method results in the lowest nMAE.", 
    digits = 2, linesep = "", label = "winners-all-methods", escape = FALSE, align = "lccccccccccccc"
  ) %>%
    # Add the top row for multi-column headers
    add_header_above(c(
      " " = 3, "Adaptive" = 2, " " = 1, "Proportion treated" = 5, " " = 1, "Effect" = 2
    ))%>%
    # Wrap the table with \resizebox{\textwidth}{!}{}
    kable_styling(latex_options = "scale_down")
  
  combined_results_latex
}

create_mean_std_table <- {
  
  # metric = "NMAE Treated Only"
  metric = "NMAE"
  
  calculate_mean_std <- function(results) {
    results %>%
      group_by(Method) %>% 
      summarise(avg = mean(!!sym(metric)), sd = sd(!!sym(metric))) %>% 
      mutate(rank =sprintf("%.3f", rank(avg)/100)) %>%
      gather(key,mean_std,avg,sd) %>%
      mutate(mean_std = ifelse(key=="sd", paste0("(",round(mean_std,2),")"),round(mean_std,2))) %>%
      mutate(rank = paste0(rank, "_", key)) %>%
      mutate(Method = ifelse(key=="sd", paste0(Method,"-sd"),Method)) %>% 
      select(-key) 
  }
  
  # Helper function to generate the results frame for each unit
  generate_unit_df <- function(unit_name) {
    unit_data <- results %>% filter(Unit == unit_name) %>%
      # Check if the global metric variable is "NMAE" and exclude specific methods
      filter(!(metric == "NMAE" & Method %in% c("DID", "DC-PR", "MC-NNM", "Mean Effect", "Median Effect", "KRLS", "GANITE")))
    
    # Calculate means and std for different conditions
    overall_stats <- calculate_mean_std(unit_data)
    empty_df = overall_stats %>% mutate(empty="") %>% select(Method, empty)
    adaptive_no <- calculate_mean_std(filter(unit_data, `Adaptive Treatment` == FALSE))
    adaptive_yes <- calculate_mean_std(filter(unit_data, `Adaptive Treatment` == TRUE))
    frac_0_05 <- calculate_mean_std(filter(unit_data, `Fraction Treated` == 0.05))
    frac_0_25 <- calculate_mean_std(filter(unit_data, `Fraction Treated` == 0.25))
    frac_0_50 <- calculate_mean_std(filter(unit_data, `Fraction Treated` == 0.5))
    frac_0_75 <- calculate_mean_std(filter(unit_data, `Fraction Treated` == 0.75))
    frac_1_0 <- calculate_mean_std(filter(unit_data, `Fraction Treated` == 1.0))
    effect_add <- calculate_mean_std(filter(unit_data, `Treatment Effect Function` == "additive"))
    effect_mult <- calculate_mean_std(filter(unit_data, `Treatment Effect Function` == "multiplicative"))
    
    # Create the final results frame by joining all means and stds based on methods
    final_table <- overall_stats %>% select(Method, mean_std, rank) %>% rename(All = mean_std) %>%
      left_join(empty_df %>% rename(empty1 = empty)) %>%
      left_join(adaptive_no %>% select(Method, mean_std) %>% rename(Adaptive_No = mean_std), by = "Method") %>%
      left_join(adaptive_yes %>% select(Method, mean_std) %>% rename(Adaptive_Yes = mean_std), by = "Method") %>%
      left_join(empty_df %>% rename(empty2 = empty)) %>%
      left_join(frac_0_05 %>% select(Method, mean_std) %>% rename(`0.05` = mean_std), by = "Method") %>%
      left_join(frac_0_25 %>% select(Method, mean_std) %>% rename(`0.25` = mean_std), by = "Method") %>%
      left_join(frac_0_50 %>% select(Method, mean_std) %>% rename(`0.50` = mean_std), by = "Method") %>%
      left_join(frac_0_75 %>% select(Method, mean_std) %>% rename(`0.75` = mean_std), by = "Method") %>%
      left_join(frac_1_0 %>% select(Method, mean_std) %>% rename(`1.0` = mean_std), by = "Method") %>%
      left_join(empty_df %>% rename(empty3 = empty)) %>%
      left_join(effect_add %>% select(Method, mean_std) %>% rename(Additive = mean_std), by = "Method") %>%
      left_join(effect_mult %>% select(Method, mean_std) %>% rename(Multiplicative = mean_std), by = "Method")
    
    final_table
  }
  
  # Generate the results for each unit
  snap_df <- generate_unit_df("CLIENT_ZIP") %>% arrange(rank) %>% select(-rank) %>% slice(1:10) 
  state_df <- generate_unit_df("State Name") %>% arrange(rank) %>% select(-rank) %>% slice(1:10) 
  county_df <- generate_unit_df("County Name") %>% arrange(rank) %>% select(-rank) %>% slice(1:10) 
  
  # Add the LaTeX formatted strings to the first entry of each block in the Method column
  snap_df$Method[1] <- paste0("\\addlinespace SNAP &&&&&&&\\\\ \\cline{1-1} \\addlinespace ", snap_df$Method[1])
  state_df$Method[1] <- paste0("\\addlinespace State &&&&&&&\\\\ \\cline{1-1} \\addlinespace ", state_df$Method[1])
  county_df$Method[1] <- paste0("\\addlinespace County &&&&&&&\\\\ \\cline{1-1} \\addlinespace ", county_df$Method[1])
  
  # Combine the results frames for all units
  combined_df <- bind_rows(snap_df, state_df, county_df) 
  combined_df[is.na(combined_df)] <- ""
  combined_df$Method[grepl("-sd",combined_df$Method)] <- ""
  colnames(combined_df) <- c("", "All", "", "Y", "N", "", "0.05", "0.25", "0.50", "0.75", "1.0", "", "Add.", "Mult.")

  
  # Create the LaTeX table with escape = FALSE to allow LaTeX commands to pass through
  combined_results_latex <- kable(
    combined_df, "latex", booktabs = TRUE, 
    caption = "Average nMAE across methods. Standard deviations shown in parentheses.", 
    digits = 2, linesep = "", label = "avg_nmae", escape = FALSE, align = "lccccccccccccc"
  ) %>%
    # Add the top row for multi-column headers
    add_header_above(c(
      " " = 3, "Adaptive" = 2, " " = 1, "Proportion treated" = 5, " " = 1, "Effect" = 2
    )) %>%
    # Wrap the table with \resizebox{\textwidth}{!}{}
    kable_styling(latex_options = "scale_down")
  
  combined_results_latex
}



