
#load packages
library(tidyverse)
library(cowplot)

library(MetBrewer)
library(ggridges)

#load data
human_solves_all_roles=data.table::fread("/mnt/home/berger/hdx/data_to_share/solves05142024.csv")
users=data.table::fread("/mnt/home/berger/hdx/data_to_share/user_data05142024.csv")

filtered_cases=data.table::fread("/mnt/home/berger/hdx/data_to_share/selected_case_ids.csv") |> pull(V1)
cases=data.table::fread("/mnt/home/berger/hdx/data_to_share/cases_data05142024.csv")

cases=separate_longer_delim(cases, cols = "sctids", delim=",")
cases$sctids=gsub("\\{|\\}", "", cases$sctids)
cases$sctids=gsub(" ", "", cases$sctids)
cases$sctids=as.character(cases$sctids)


cases=cases |> rename(pc_id=id)

case_correct_sctids=cases |> select(pc_id, sctids) |> rename(correct_sctid=sctids)
case_correct_sctids=case_correct_sctids |> distinct()


human_solves_all_roles=human_solves_all_roles |> filter(rank<=5)
human_solves_all_roles$pc_id=as.factor(human_solves_all_roles$pc_id)


user_more_5_all_roles=human_solves_all_roles |> distinct(pc_id,solver_id) |> group_by(solver_id)|> summarise(n=n()) |> 
  filter(n>4) |> pull(solver_id) |> unique()

human_solves_all_roles=human_solves_all_roles |> filter(pc_id %in% filtered_cases,
                                                        solver_id %in% user_more_5_all_roles)


results_role=users |> select("_id",role_name) |> rename(solver_id="_id") |> 
  right_join(human_solves_all_roles, by="solver_id") |> 
  filter(role_name!="Intern") |> 
  left_join(case_correct_sctids, by="pc_id", relationship = "many-to-many") |> 
  select(pc_id,solver_id,role_name,rank,sctid,correct_sctid) |> 
  mutate(correct=ifelse(sctid==correct_sctid,1,0)) |>
  mutate(rank_correct=ifelse(correct==1,rank,1000)) |> 
  ungroup() |> 
  group_by(solver_id,pc_id) |> 
  slice_min(rank_correct, n=1, with_ties = FALSE) |> 
  mutate(rr=ifelse(rank_correct==1000,0,
                   1/rank)) |> 
  group_by(solver_id,role_name) |> 
  summarise(mrr=mean(rr))  |> 
  mutate(role_name=ifelse(role_name=="Med student","Student",role_name))


comparison_levels=results_role |> 
  mutate(role_name=factor(role_name, levels=c("Student","Resident","Fellow","Attending"))) |> 
  ggplot(aes(x=mrr,y=role_name, color=role_name,fill=role_name))+
  stat_density_ridges(quantile_lines = TRUE, quantiles = 2, color="black", alpha=.76)+
  #stat_halfeye(point_color="black",point_interval = "median_qi",interval_colour="black")+
  ylab("")+
  xlab("MRR")+
  theme_minimal()+
  theme(panel.border = element_rect(color="black", fill=NA))+
  theme(legend.position = "none")+
  scale_fill_met_d("Cassatt1")
comparison_levels


results_comp_experts_students=users |> select("_id",role_name) |> rename(solver_id="_id") |> 
  right_join(human_solves_all_roles, by="solver_id") |> 
  filter(role_name!="Intern") |> 
  mutate(role_name=ifelse(role_name=="Med student","Student","Experienced")) |> 
  left_join(case_correct_sctids, by="pc_id", relationship = "many-to-many") |> 
  select(pc_id,solver_id,role_name,rank,sctid,correct_sctid) |> 
  mutate(correct=ifelse(sctid==correct_sctid,1,0)) |>
  mutate(rank_correct=ifelse(correct==1,rank,1000)) |> 
  ungroup() |> 
  group_by(solver_id,pc_id) |> 
  slice_min(rank_correct, n=1, with_ties = FALSE) |> 
  mutate(rr=ifelse(rank_correct==1000,0,
                   1/rank)) |> 
  group_by(pc_id,role_name) |> 
  summarise(mrr=mean(rr)) |> 
  pivot_wider(names_from = role_name,values_from = mrr) |> 
  drop_na() |> 
  group_by(pc_id) |> 
  mutate(diff=Experienced-Student)

comparison_experts_students=results_comp_experts_students |> 
  ggplot(aes(x=diff))+
  geom_histogram(binwidth = .05, fill="#829FDF")+
  geom_vline(xintercept = 0)+
  ylab("Count of cases")+
  xlab("MRR difference between physicians and students on a case")+
  theme_minimal()+
  theme(panel.border = element_rect(color="black", fill=NA))+
  theme(legend.position = "none")
comparison_experts_students


plot_grid(comparison_levels, comparison_experts_students, labels = c("a","b"), ncol = 2) |> 
  ggsave2(filename="comparison_tenure.pdf",height = 5,width = 10)


