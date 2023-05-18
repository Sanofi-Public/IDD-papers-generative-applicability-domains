for set in "oxathiazines_11bHSD" "full_11bHSD" "chembl_11bHSD" "renin" "jak2" "jak2_smiles_ga" "jak2_graph_ga" "ureas_11bHSD"
	do jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=300 --allow-errors --execute "analysis_"$set".ipynb"
	#ls -l "analysis_"$set".ipynb"
       	done

