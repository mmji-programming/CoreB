from CoreB import Core, Agent, Controller
import time


# Simple Sample 
event = Controller(name="event")

def func(name, size=6, delay=1):
    
    for i in range(size):
        
        if event.is_pause():
            event.wait_until_pause()
        
        print(f"{i + 1} from {name}")
        time.sleep(delay)
    
    return [name, 1, 2, 3]


tasks = [Agent(func, name="func1", size=10, _return_name="func1"), Agent(func, name="func2", _return_name="func2")]
core = Core(list_of_agents=tasks)
core.set_controller(event)
core.run()

time.sleep(3)
core.pause("event")

for i in range(3):
    print(f"Sleep {3 - i}", end="\r")
    time.sleep(1)

core.add_task(Agent(func, name="func3 added", _return_name="func3"))

time.sleep(7)

core.resume("event")

time.sleep(9)
print(core.returns)