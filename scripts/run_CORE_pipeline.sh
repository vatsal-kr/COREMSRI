#!/bin/bash
DEBUGDIR=debug/CQPyUs_results
DATASET=dataset/CQPyUs
QUERY="Redundant comparison"

# Stage 2 and 3
uv run src/run_llm_proposer.py -e $DATASET -t $DEBUGDIR/proposer_results -r $DATASET -s $DEBUGDIR/files.pkl --Queries "$QUERY" --model='gpt-4o-mini'

# Stage 4
uv run src/run_codeql_verifier.py -i $DEBUGDIR/proposer_results -c codeql-home/codeql-repo/python/ql/src/ -d $DEBUGDIR/verifier_results/db/ -o $DEBUGDIR/verifier_results/res/ --Queries "$QUERY" 

uv run src/run_codeql_analysis.py -i $DATASET -g $DEBUGDIR/proposer_results/ -d $DEBUGDIR/verifier_results/db/ -r $DEBUGDIR/verifier_results/res/ -o $DEBUGDIR/analysis_results/ -s $DEBUGDIR/files.pkl --Queries "$QUERY" 

uv run src/get_diffs.py -p $DATASET -g $DEBUGDIR/proposer_results -a $DEBUGDIR/analysis_results/demo_filefix_types_Top10.pickle -s $DEBUGDIR/diff_folder
# Stage 5
uv run src/run_llm_ranker.py -d $DEBUGDIR/diff_folder -o $DEBUGDIR/ranker_results/ --Queries "$QUERY" --model='gpt-4o-mini'

uv run src/get_results.py -r $DEBUGDIR/ranker_results/ -p $DATASET -g $DEBUGDIR/proposer_results/ -d $DEBUGDIR/diff_folder/ -o $DEBUGDIR/results/ -j $DEBUGDIR/results.json