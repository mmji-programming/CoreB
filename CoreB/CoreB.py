from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from functools import lru_cache
from threading import Event
from waiting import wait
from math import log
import traceback
import ctypes
import time
import os


# Contact: mmji-programming@proton.me
# Core execution for concurrently blocking functions
# TODO: Logic optimization and implementation



# Special Errors
class DuplicateKeyError(Exception):
    """Found a key that already exists"""
    
    __module__ = Exception.__module__

class NoSetController(Exception):
    """No controller is configured with this name."""
    
    __module__ = Exception.__module__



# Main classes
class Time:
    
    def __init__(self) -> None:
        """Capture current monotonic time"""
        self.now = time.monotonic()
    
    def passed(self) -> float:
        """Elapsed time"""
        
        return time.monotonic() - self.now

    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        
        return f"{self.__get_name()}(Start at: {self.now} [monotonic])"
    
class Queue:
    
    """
        Queue with execution time O(1).
        
        This queue uses a dictionary to store values.
        Based on the fact that the keys are a unique and sequential integer and their values ​​are the desired item.
        
        Two counters named __enqueue_counter and __dequeue_counter are used to add or remove values ​​from the queue, 
            check whether the queue is full or empty and control the capacity of the queue if it is defined.
    """
    
    def __init__(self, capacity: int = -1) -> None:
        
        self.size = 0 # save size
        self.__queue: dict[int, Any] =  dict() # Main Queue
        self.__enqueue_counter = 0 # count enqueue
        self.__dequeue_counter = 0 # count dequeue
        self.__set_capacity = capacity > -1 # having a specific or infinite capacity
        self.capacity = capacity
        self.empty = True

    def enqueue(self, item: Any) -> None:
        """Add an item to the queue."""
        
        if self.__set_capacity and self.__enqueue_counter >= self.capacity:
            return
        
        self.size += 1
        self.empty = False
        self.__queue[self.__enqueue_counter] = item
        self.__enqueue_counter += 1
    
    def dequeue(self) -> Any:
        """Remove an item from the queue."""
        
        if self.__set_capacity and self.__dequeue_counter >= self.capacity:
            self.empty = True
            return None # Queue is empty

        item = self.__queue.pop(self.__dequeue_counter, None)
        
        if self.__dequeue_counter + 1 < self.__enqueue_counter:
            self.__dequeue_counter += 1
        
        else:
            self.__enqueue_counter = 0
            self.__dequeue_counter = 0
            self.empty = True
                
        return item
    
    def get(self, key: int) -> Any | None:
        return self.__queue.get(key, None)
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return f"{self.__get_name()}({self.__queue}, {'inf' if self.capacity == -1 else self.capacity})"
 
class Lable:
    """
        Labels for each event
        
        Labels:
            INIT_CORE
            START_CORE
            END_CORE
            ADD_AGENT
            ADD_AGENT_RUNNING: This label is used when an Agent is added to the core during execution.
            START_AGENT
            END_AGENT
            EXCEPTION_AGENT
            EXECPTION: Occurrence of internal errors that are not specific to the Agent.
            START_INJECTION
            END_INJECTION
            INJECT: Executing Agents.
            TERMINATE: Termination the execution of the core due to an error in an Agent or as requested.
            FORCE_TERMINATE
            SET_CONTROLLER
            UPDATE_COUNTERS
            WAIT_CONTROLLER
            TERMINATE_CONTROLLER
            PAUSE_CONTROLLER
            RESUME_CONTROLLER
    """
    
    def __init__(self, _object: Any, lable: str, *args, **kwargs) -> None:

        self.object = _object
        self.lable = lable
        self.args = args
        self.kwargs = kwargs
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        
        try:
            return f"{self.__get_name()}(Object: {repr(self.object.__str__())}, Lable: {repr(self.lable)}" + (f", {', '.join(str(i) for i in self.args)}" if self.args else "") + (f", {', '.join(f'{key}: {repr(value)}' for key, value in self.kwargs.items())}" if self.kwargs else "") + ")"
        
        except:
            return f"{self.__get_name()}(Object: {self.object}, Lable: {repr(self.lable)})"

class ExecutionTrack:
    """Storing events sequentially"""
    
    events = Queue(capacity=-1)
    number_of_core = 0
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        
        return f"{self.__get_name()}(NumberOfCore: {self.number_of_core}, Events: {self.events.__str__()})"

class Controller:
    """
        Create a 'threading.Event' for one or a set of functions to control them when the main loop of the program is executed.
        Note: A name must be set for the controller.
        
        For example, you create a controller for a set of functions and another distinct control for another set of functions, 
            and you have more control over the execution of functions during the program.
        
        You can define one or more global controllers and use them directly in your function, 
            or you can define one or more controllers and give them as input to your functions with a desired name (event is suggested).
    """
    
    def __init__(self, name: str) -> None:
        
        self.__terminate = False
        self.name = name
        self.__event = Event()
        self.__event.set() # Set internal flag to true
    
    def terminate(self) -> None:
        """Terminate the execution of functions that use this controller"""
        
        self.__terminate = True
    
    def is_terminate(self) -> bool:
        return self.__terminate
    
    def wait_until_pause(self, timeout: float = None) -> None:
        """
            As long as the control of the main loop is held, the execution of the function pause.
            When the timeout argument is present and not None, it should be a floating point number specifying a timeout for the operation in seconds (or fractions thereof).
        """
        
        self.__event.wait(timeout=timeout)
    
    def pause(self) -> None:
        """Pause execution of one or a set of functions by this controller."""
        
        self.__event.clear()
    
    def resume(self) -> None:
        """Resume execution of one or set of functions by this controller."""
        
        self.__event.set()
    
    def is_pause(self) -> bool:
        """Returns True if the controller is paused, otherwise returns False."""
        
        return not self.__event.is_set()
    
    def __get_name(self) -> str:
        return self.__class__.__name__ # Controller
    
    def __str__(self) -> None:
        return f"{self.__get_name()}(Name: {self.name})"
    
class Return:
    """
        Saving return values ​​from all functions during Core execution.
        
        Note: that if you use a function multiple times in Core, 
              you must define a different name for each of them in Agent so that their outputs are saved correctly if there is a difference.
            
        
    """
        
    __core_done_counter = {
        "size": 0,
        "counter": 0,
        "workers_added": 0,
        "done_counter": 0
    } # Saving the number of tasks and completed tasks as well as managing the number of workers in Core.__executor
        
    returns = {} # Store return values
    
    def __init__(self, core_name: str, *args: Any, _set_name: str = "", **kwargs) -> None:
        
        if not self.returns[core_name].get(_set_name, False):
            self.returns[core_name][_set_name] = {"exec_numb": 0}
        
        self.returns[core_name][_set_name]["exec_numb"] += 1
        exec_numb = self.returns[core_name][_set_name]["exec_numb"]
        
        if kwargs:
            self.returns[core_name][_set_name][repr(exec_numb)] = {
                "kwargs": kwargs,
                "args": args[0] if len(args) == 1 else args
            }
        
        else:
            self.returns[core_name][_set_name][repr(exec_numb)] = args[0] if len(args) == 1 else args
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        
        return f"{self.__get_name()}(Parameters({self.__core_done_counter}), ReturnsLength({len(self.returns)}))"
     
class Agent:
    """
        The functions are not directly executed on the Core, but we have to create an Agent from them to execute them on the Core.
        
        Agent's task is to add a series of functions, including storing return values ​​from functions in Return, 
            controlling the output, creating a future and coroutine from the same future to be executed simultaneously and 
            managing the number of workers in the main Core.
    
        Each Agent has a name that is used to store the output of the function and differentiate between the outputs.
            Note: Note that if you don't set a distinct name for Agent, it will take the name of its function. [_set_name]
            Note: You can also use a special name that is only used to save the output. [_return_name]
        
        
        Usage:
            Agent(function, args, kwargs)  
        
        special_kwargs:    
            _return_name: Setting a special name that is only used to save the output.
            _set_name: Setting a specific name as the default name of the Agent, which takes the name of its input function by default.
            _use_cache: Configure cache usage for the Agent. [default: False]
            _cache_size: Determine cache size. [default: 128]
            _cache_typed: If typed is True, arguments of different types will be cached separately. [default: False]

    """
    
    def __init__(self, function: Callable, *args, **kwargs) -> None: 
        
        # Self-Agent init
        self.name = kwargs.pop("_set_name", function.__name__)
        self.__return_name = kwargs.pop("_return_name", self.name)
        
        # Agent-Function init
        self.__use_cache = kwargs.pop("_use_cache", False)
        self.__cache_size = kwargs.pop("_cache_size", 128)
        self.__typed = kwargs.pop("_cache_typed", False)

        
        if self.__use_cache:
            self.function = self.__wrapFunction(self.__use_cache_(function, maxsize=self.__cache_size, typed=self.__typed))
        else:
            self.function = self.__wrapFunction(function)

        self.args = args
        self.kwargs = kwargs
    
    def __use_cache_(self, function: Callable, maxsize: int | None, typed: bool):
        return lru_cache(maxsize=maxsize, typed=typed)(function)
    
    def __wrapFunction(self, function: Callable) -> Callable:
        """Add necessary functionality to the function."""
        
        def wrapper(*args, **kwargs):
            
            core_name = getattr(self, "__core_name")            
            ExecutionTrack.events.enqueue(Lable(self, "START_AGENT", core_name=core_name))
            start = Time()
                        
            try:
            
                result = function(*args, **kwargs)
                
                if self.__use_cache:                    
                    function.cache_clear()
                    
                Core._Core__executors[core_name]["size"] -= 1
                Return._Return__core_done_counter["counter"] += 1
                
                _exec_time = start.passed()        
                ExecutionTrack.events.enqueue(Lable(self, "END_AGENT", exec_time=_exec_time, _return=result, core_name=core_name))

                # Store the return value of the function in the Return class
                Return(core_name, result, _set_name=self.__return_name, exec_time=_exec_time)
                
                # Management of the number of workers based on the total number of tasks in the Core class            
                if (Return._Return__core_done_counter["size"] == Return._Return__core_done_counter["counter"]):

                    Return._Return__core_done_counter["done_counter"] += 1
                    ExecutionTrack.events.enqueue(Lable(core_name, "END_CORE", last_agent=self.name))
                    
                    # Decrease _max_workers
                    Core._Core__executors[core_name]["executor"]._max_workers -= Return._Return__core_done_counter["workers_added"]
                                    
                    # Reset size & counters & workers_added
                    Return._Return__core_done_counter["size"] = 0
                    Return._Return__core_done_counter["counter"] = 0
                    Return._Return__core_done_counter["workers_added"] = 0
                    Core._Core__executors[core_name]["size"] = 0
                    ExecutionTrack.events.enqueue(Lable("Return.__core_done_counter", "UPDATE_COUNTERS"))
                    
                    # Log for core
                    Return.returns[core_name]["exec_time"] += Core._Core__cores_time[core_name]["time"].passed()
                    Core._Core__cores_time[core_name]["time"] = Time()
                    
            except Exception as e:
                
                _exec_time = start.passed()
                Return(core_name, str(e), _set_name=self.__return_name, exec_time=_exec_time)
                ExecutionTrack.events.enqueue(Lable(self, "EXCEPTION_AGENT", error=e, exec_time=_exec_time, _return=None, core_name=core_name))

                # show error and terminate core and reset values
                print(traceback.format_exc())          
                
                ExecutionTrack.events.enqueue(Lable(core_name, "TERMINATE", error=e, by=f"{self.__get_name()}({repr(self.name)})"))
                
                if self.__use_cache:
                    function.cache_clear()
                    
                # Core termination
                Core._Core__terminators[core_name]()
                
                raise e
                            
        return wrapper
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        
        content = self.__get_name() + "("
        content += f"Name: {repr(self.name)}, "
        content += f"_return_name: {repr(self.__return_name)}, "
        
        if self.__use_cache:
            content += f"_use_cache: {self.__use_cache}, "
            content += f"_cache_size: {self.__cache_size}, "
            content += f"_cache_typed: {self.__typed}, "
        
        content += f"FuncArgs: {self.args}, "
        content += f"FuncKwargs: {self.kwargs}"

        content += ")"

        return content
            
class Loop:
    """
        Loop is responsible for receiving agents and dropping them on the core.
    """
    
            
    def __init__(self, *agents: Agent, excecutor: 'ThreadPoolExecutor', core_name: str, list_of_agents: list[Agent] = []) -> None:
        
        self.__agents = list(agents) + list_of_agents
        self.size = len(self.__agents) # Saving the number of tasks (we don't need it now)
        self.excecutor = excecutor
        self.core_name = core_name

    def injection(self, name: str) -> None:
        
        ExecutionTrack.events.enqueue(Lable(name, "START_INJECTION"))

        while self.size:
            
            agent = self.__agents.pop(0) # Get an Agent
            setattr(agent, "__core_name", name) # Set core name for agent
            
            ExecutionTrack.events.enqueue(Lable(agent, "INJECT", core_name=name)) # Log
            
            self.excecutor.submit(agent.function, *agent.args, **agent.kwargs) # Add Agent to thread
            self.size -= 1 # Update size

        ExecutionTrack.events.enqueue(Lable(name, "END_INJECTION"))
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return f"{self.__get_name()}(Core: {self.core_name})"
        
class Core:
    
    """
        The main core that is responsible for receiving, executing and managing functions. 
    """
    
    # These two variables exist throughout the entire runtime of Core.
    # __executor = ThreadPoolExecutor() # The main core that is responsible for receiving, executing and managing functions.
    __controllers: dict[str, Controller] = {} # Saving controllers and using them throughout the program.
    __executed_once_core = {}
    __executors = {}
    __terminates = {}
    __cores_time = {}
    __terminators = {}
    
        
    def __init__(self, *agents: Agent, list_of_agents: list[Agent] = [], name: str = "") -> None:

        self.__executor = ThreadPoolExecutor()
        self.__agents: list[Agent] = list(agents) + list_of_agents
        self.__agents_size = len(self.__agents) 
        
        # Specify a unique name for each core
        if (_id := id(self)) not in self.__executed_once_core: # Safe exist check
            
            ExecutionTrack.events.enqueue(Lable(self, "INIT_CORE"))
            
            self.__executed_once_core[_id] = True
            self.__core_name = name if name else f"__core_{ExecutionTrack.number_of_core}"
            ExecutionTrack.number_of_core += 1
            self.__terminates[self.__core_name] = False
            self.__executors[self.__core_name] = {"executor": self.__executor, "size": 0, "workers_added": 0}
            self.__cores_time[self.__core_name] = {"time": Time()}
            self.__terminators[self.__core_name] = self.force_terminate
            Return.returns[self.__core_name] = {"exec_numb": 0, "exec_time": 0}
        
        
        
        # Increase _max_workers in self.__executor
        self.__workers_size = self.calculate_workers(self.__agents_size)
        self.__executors[self.__core_name]["executor"]._max_workers += self.__workers_size
        
        Return._Return__core_done_counter["size"] += self.__agents_size
        self.__executors[self.__core_name]["size"] += self.__agents_size
        
        Return._Return__core_done_counter["workers_added"] += self.__workers_size
        self.__executors[self.__core_name]["workers_added"] += self.__workers_size
        
        # Init Loop
        self.loop = Loop(list_of_agents=self.__agents, excecutor=self.__executors[self.__core_name]["executor"], core_name=self.__core_name) # Set loop
        self.returns = Return.returns # Pointing to the place where the outputs are stored, which of course can be accessed directly from the Return class.
        
    def run(self):
        """
            Run Core.
            Every time it is executed, it injects existing agents into the loop.
        """
        
        if not self.__terminates[self.__core_name]:
            
            Return.returns[self.__core_name]["exec_numb"] += 1
            ExecutionTrack.events.enqueue(Lable(self, "START_CORE"))
            self.loop.injection(self.__core_name)
            
    def set_controller(self, controller: Controller) -> None:
        """Set the controller with their name in Core.__controllers"""
        
        if self.__controllers.get(controller.name, None):
            ExecutionTrack.events.enqueue(Lable(self, "EXCEPTION", exception="This controller has already been added."))
            raise DuplicateKeyError("This controller has already been added.")
        
        ExecutionTrack.events.enqueue(Lable(self, "SET_CONTROLLER", repr(controller.__str__())))
        self.__controllers[controller.name] = controller
        
    def __check_controller(self, controller_name: str) -> None:
        """Checking the absence of the controller."""
        
        if controller_name not in self.__controllers:
            ExecutionTrack.events.enqueue(Lable(self, "EXCEPTION", exception="A controller with this name has not been configured."))
            raise KeyError("A controller with this name has not been configured.")
    
    def terminate(self, controller_name: str) -> None:
        """Terminate running the controller"""

        self.__check_controller(controller_name=controller_name)
        ExecutionTrack.events.enqueue(Lable(controller_name, "TERMINATE_CONTROLLER", core_name=self.__core_name))
        self.__controllers[controller_name].terminate()
        
    def is_terminate(self, controller_name: str) -> bool:
        """Terminated or not terminated status [It is usually used inside the code piece being developed.]"""
        
        self.__check_controller(controller_name=controller_name)
        return self.__controllers[controller_name].is_terminate()
            
    def wait_until_pause(self, controller_name: str, timeout: float = None) -> None:
        """Temporarily stops functions that use a shared controller. [It is usually used inside the code piece being developed.]"""
        
        self.__check_controller(controller_name=controller_name)
        ExecutionTrack.events.enqueue(Lable(controller_name, "WAIT_CONTROLLER", core_name=self.__core_name))
        self.__controllers[controller_name].wait_until_pause(timeout=timeout)
        
    def pause(self, controller_name: str) -> None:
        """Changing the internal flag of the controller to true to create a pause in the functions."""
        
        self.__check_controller(controller_name=controller_name)
        
        ExecutionTrack.events.enqueue(Lable(controller_name, "PAUSE_CONTROLLER", core_name=self.__core_name))
        self.__controllers[controller_name].pause()
        
    def resume(self, controller_name: str) -> None:
        """Changing the internal flag of the controller to false and resuming the execution of functions."""
            
        ExecutionTrack.events.enqueue(Lable(controller_name, "RESUME_CONTROLLER", core_name=self.__core_name))
        self.__controllers[controller_name].resume()
    
    def is_pause(self, controller_name: str) -> bool:
        """Checking the issuance of the pause command. [It is usually used inside the code piece being developed.]"""
        
        self.__check_controller(controller_name=controller_name)
        return not self.__controllers[controller_name].is_pause()

    def add_task(self, *agents: Agent, list_of_agents: list[Agent] = []) -> None:
        """Adding tasks while Core is running and injecting them into the Loop"""
        
        if not self.__terminates[self.__core_name]:
        
            ExecutionTrack.events.enqueue(Lable(self, "ADD_AGENT_RUNNING", *(list(agents) + list_of_agents)))
            self.__init__(*agents, list_of_agents=list_of_agents)
            self.run()
         
    def force_terminate(self) -> None:
                
        if not self.__terminates.get(self.__core_name, False):
            
            Return._Return__core_done_counter["done_counter"] += 1
            self.__terminates[self.__core_name] = True
        
                
            executor_object = self.__executors[self.__core_name]
            threads = executor_object["executor"]._threads
            
            # Decrease workers
            size = executor_object["size"]
            workers_added = executor_object["workers_added"]
            
            # Reset
            self.__executors[self.__core_name].update({
                "size": 0,
                "workers_added": 0,
                "executor": None
            })
            
            Return._Return__core_done_counter["size"] -= size
            Return._Return__core_done_counter["workers_added"] -= workers_added
            Return.returns[self.__core_name]["exec_time"] += self.__cores_time[self.__core_name]["time"].passed()
            self.__cores_time[self.__core_name]["time"] = Time()
            
            
            ExecutionTrack.events.enqueue(Lable(self, "UPDATE_COUNTERS"))
            
            # Stop core
            for thread in threads:
                
                _id = thread.native_id # Get thread id
                result = ctypes.pythonapi.PyThreadState_SetAsyncExc(_id, ctypes.py_object(SystemExit)) # Raise exception
                
                # Exception raise failure
                if result > 1:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(_id, 0)

    @classmethod
    def force_terminate_Core(self) -> None:
        
        ExecutionTrack.events.enqueue(Lable(str(Core), "FORCE_TERMINATE"))
        
        # Termination all cores
        for core_name in self.__terminators:
            self.__terminators[core_name]()
            ExecutionTrack.events.enqueue(Lable(core_name, "TERMINATE"))
    
    @classmethod 
    def is_end(self) -> bool:
        
        if Return._Return__core_done_counter["done_counter"] == ExecutionTrack.number_of_core:
            
            Return._Return__core_done_counter.update({
                "size": 0,
                "counter": 0,
                "workers_added": 0,
                "done_counter": 0
            })
            
            return True

        return False

    @classmethod
    def wait_until_end(self, period: float = 0.9) -> None:
        wait(Core.is_end, sleep_seconds=period)
    
    @classmethod
    def calculate_workers(self, size: int) -> int:
        """
            An unprincipled way to calculate the number of required workers based on the number of tasks and the number of CPU cores.
            
            TODO: Creating a formula based on the number of tasks, their type, CPU conditions and important and basic parameters.
        """

        if size <= 11:
            return 0
                
        return int(log(size / 10, 1.101)) * os.cpu_count()
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
    
        return f"{self.__get_name()}(Name: {repr(self.__core_name)})"  

