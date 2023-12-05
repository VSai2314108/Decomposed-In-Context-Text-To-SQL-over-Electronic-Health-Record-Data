# run in the long environment
import subprocess
import os
import pandas as pd
from tqdm import tqdm

def create_text_sql(csv_file, directory):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Split the DataFrame into two Series for predicted and gold SQLs
    predicted_sqls = df['PREDICTED SQL']
    gold_sqls = df['GOLD SQL']
    db_ids = df['DATABASE']
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write the predicted SQLs to a text file
    with open(f'{directory}/predicted_SQLS.txt', 'w') as file:
        for sql in predicted_sqls:
            file.write(sql + '\n')
    
    # Write the gold SQLs to a text file
    with open(f'{directory}/Gold_SQLS.txt', 'w') as file:
        for db_id,sql in zip(db_ids,gold_sqls):
            file.write(sql + '\t' + db_id +'\n')
            
def load_results(directory):
    with open(directory+"/predicted_SQLS.txt", 'r') as file:
        predicted_sqls = file.readlines()
    with open(directory+"/Gold_SQLS.txt", 'r') as file:
        gold_sqls = file.readlines()
    sqls = []
    for gold_sql, predicted_sql in zip(gold_sqls, predicted_sqls):
        sqls.append([gold_sql.split("\t")[1].strip(), gold_sql.split("\t")[0].strip(), predicted_sql.strip()])
    return sqls

def get_accuracy(db_id, gold_sql, predicted_sql):
    with open('test-suite-sql-eval-master/Gold_test.txt', 'w') as f:
        f.write(gold_sql + "\t" + db_id)
    with open('test-suite-sql-eval-master/Predicted_test.txt', 'w') as f:
        f.write(predicted_sql)
    cmd_str = "python3 test-suite-sql-eval-master/evaluation.py --gold test-suite-sql-eval-master/Gold_test.txt --pred test-suite-sql-eval-master/Predicted_test.txt --db test-suite-sql-eval-master/database/ --etype exec "
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    os.remove("test-suite-sql-eval-master/Gold_test.txt")
    os.remove("test-suite-sql-eval-master/Predicted_test.txt")
    print(result)
    acc = float(result.stdout[-21:-16])
    return acc

if __name__ == '__main__':
    directory = "Results/SQLChainWithSchema_linking"
    create_text_sql("LLAMA-RESULTS-CSV.csv", directory=directory)
    directory_list = load_results(directory)
    results = []
    for entry in directory_list:
        print(entry)
        acc = get_accuracy(entry[0], entry[1], entry[2])
        decision = "CORRECT" if acc == 1 else "WRONG"
        results.append([entry[0], entry[1], entry[2], decision])
    df = pd.DataFrame(results, columns=['DATABASE', 'GOLD SQL', 'PREDICTED SQL', 'DECISION'])
    df.to_csv("/blue/cap4773/somasundaramv/Few-shot-NL2SQL-with-prompting/Analysis/analysis_of_results.csv", index=False)