# CoreB

![CoreB Logo](CoreB/img/coreb.png)

CoreB is a library for managing concurrent function execution and control in Python applications. This library includes capabilities for managing **ThreadPoolExecutor**, controlling processes with **Controller**, handling function outputs with **Return**, and executing tasks with **Agent**.

## Features ✨
- **Manage concurrent function execution** using **ThreadPoolExecutor**
- **Pause and resume function execution** with **Controller**
- **Handle function outputs** with **Return**
- **Manage and execute tasks in the processing core** with **Agent**
- **Optimized structure for executing tasks in a processing loop** with **Loop**

## Installation 📦
Run the ```pip install -r requirements.txt``` command and add the **CoreB** file to your project.

## Usage 📖
```python
import time
from CoreB import Controller, Agent, Core

# Create a controller
controller = Controller(name="event")

def func(name, size=6, delay=1):
    for i in range(size):
        if controller.is_pause():
            controller.wait_until_pause()
        print(f"{i + 1} from {name}")
        time.sleep(delay)

# Define tasks
tasks = [Agent(func, name="func1", size=10, _return_name="func1"), Agent(func, name="func2", _return_name="func2")]
core = Core(list_of_agents=tasks)
core.set_controller(controller)
core.run()

# Pause and resume execution
time.sleep(3)
core.pause("event")
core.add_task(Agent(func, name="func3 added", _return_name="func3"))
time.sleep(7)
core.resume("event")
```

## Classes and Descriptions 🔍
### 1. `Controller`
A class for controlling function execution throughout the program.

- `pause()`: Pause execution
- `resume()`: Resume function execution
- `wait_until_pause(timeout)`: Wait until the controller resumes execution
- `is_pause() -> bool`: Check the pause status

### 2. `Agent`
This class represents a function to be executed in **CoreB**.

- Takes **a function, arguments, and a unique name for storing output** as input.

### 3. `Core`
The main class responsible for executing tasks.

- `run()`: Execute added functions
- `pause(controller_name)`: Pause execution of functions associated with a specific controller
- `resume(controller_name)`: Resume execution of functions
- `add_task(*agents)`: Add new tasks while running

### 4. `Return`
Stores the outputs of executed functions in **CoreB**.

### 5. `Loop`
Manages task execution within a processing loop.

## TODO 🛠️
- [ ] Implementing advanced logging
- [ ] Implementing better methods for computing the number of **Workers** based on CPU conditions
- [ ] Add documentation for all classes and methods.
- [ ] Implement unit tests for core functionalities.
- [ ] Create examples demonstrating usage of the core.
- [ ] Optimize performance for large-scale tasks.
- [ ] Ensure compatibility with projects that have a main loop.

## License 📜
This project is released under the **MIT** license. Usage, modification, and distribution are permitted.

---
🚀 **CoreB** is a simple and powerful solution for managing concurrent processing in Python!
