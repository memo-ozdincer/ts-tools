#!/usr/bin/env python3
import sqlite3
import json

# Check multi-mode HPO database structure
for db_name, db_path in [
    ('hip_multimode', '/Users/memoozdincer/Desktop/outputs/hip_multi_mode_hpo_1820826.db'),
    ('scine_multimode', '/Users/memoozdincer/Desktop/outputs/scine_multi_mode_hpo_1809794.db'),
    ('hip_sella', '/Users/memoozdincer/Desktop/outputs/hip_hpo_job1802595.db'),
]:
    print(f'\n=== {db_name} ===')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all unique parameter names
    cursor.execute('SELECT DISTINCT param_name FROM trial_params')
    params = [r[0] for r in cursor.fetchall()]
    print(f'Parameters: {params}')
    
    # Get all unique user attribute keys
    cursor.execute('SELECT DISTINCT key FROM trial_user_attributes')
    attrs = [r[0] for r in cursor.fetchall()]
    print(f'User attrs: {attrs}')
    
    # Get best trial data
    cursor.execute('''
        SELECT t.trial_id, t.number, tv.value 
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT 3
    ''')
    print('\nTop 3 trials:')
    for row in cursor.fetchall():
        trial_id, number, value = row
        print(f'  Trial {number}: score={value:.4f}')
        
        # Get params
        cursor.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id = ?', (trial_id,))
        params_dict = {r[0]: r[1] for r in cursor.fetchall()}
        print(f'    Params: {params_dict}')
        
        # Get user attrs
        cursor.execute('SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?', (trial_id,))
        attrs_dict = {}
        for k, v in cursor.fetchall():
            try:
                attrs_dict[k] = json.loads(v)
            except:
                attrs_dict[k] = v
        print(f'    User attrs: {attrs_dict}')
    
    conn.close()
