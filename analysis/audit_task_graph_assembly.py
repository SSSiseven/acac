import json, runpy
from pathlib import Path
n = runpy.run_path('tests/test_task_graph_v1_adapter.py')
env_cls = n['Overcooked_MA_V1']
reward = {'subtask finished':10, 'correct delivery':200, 'wrong delivery':-5, 'step penalty':-0.1}
builder = n['TaskGraphBuilder'](n['TaskGraphMetadata'](grid_shape=(7,7), recipe_ingredients=('tomato','lettuce','onion'), macro_action_names=n['ACTIONS'], walkable_positions=n['WALKABLE_A7']))
def new_env():
    e = env_cls((7,7), 'lettuce-onion-tomato salad', reward, map_type='A', n_agent=3, rand_start=False)
    e.reset()
    return e
def graph(e):
    e._get_obs()
    obs = e._get_macro_obs()[0]
    return builder.build(*n['decode_v1_macro_observation'](obs,n['LAYOUT'],self_robot_id=0,self_ready=True))
def mask(g, task): return bool(g.robot_task_mask[0,g.task_ids.index(task)])
e = new_env()
a, t, k = e.agent[0], e.tomato[0], e.knife[0]
e.map[a.x][a.y] = 0; a.move(1,1); e.map[1][1] = 2
e.map[t.x][t.y] = 1; t.move(k.x,k.y); t.chopped=True; t.cur_chopped_times=3; k.hold(t)
g = graph(e)
pickup_mask = mask(g,'get:tomato')
e.run([1,0,0])
pickup_succeeded = a.holding is t
e = new_env()
a,t,l,p = e.agent[0],e.tomato[0],e.lettuce[0],e.plate[0]
e.map[a.x][a.y]=0; a.move(5,5); e.map[5][5]=2
e.map[t.x][t.y]=1; t.chopped=True; t.cur_chopped_times=3; a.pickup(t)
e.map[l.x][l.y]=1; l.chopped=True; l.cur_chopped_times=3; p.contain(l)
g = graph(e)
plate_mask = mask(g,'get:plate1')
e.run([4,0,0])
deposit_succeeded = p.containing is not None and t in p.containing
report = {'chopped_food_pickup': {'task_mask':pickup_mask, 'original_macro_succeeded':pickup_succeeded}, 'partial_plate_add_ingredient': {'task_mask':plate_mask, 'original_macro_succeeded':deposit_succeeded}}
assert not pickup_mask and pickup_succeeded, report
assert not plate_mask and deposit_succeeded, report
out=Path('data/audit/stage3_preflight.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))