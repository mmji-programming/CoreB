from concurrent.futures import ThreadPoolExecutor, Future
from typing import Coroutine, Generator, Any, Callable
from threading import Event
from math import log
import os


# Contact: mmji_programming@proton.me
#
#
# Core execution for concurrently blocking functions
# TODO: Add log
# TODO: Logic optimization and implementation
# TODO: Implementation of a mechanism to detect the end of execution of all cores. [Optional]


# Special Errors
class DuplicateKeyError(Exception):
    """Found a key that already exists"""
    
    __module__ = Exception.__module__

class NoSetController(Exception):
    """No controller is configured with this name."""
    
    __module__ = Exception.__module__



# Main classes
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
        
        self.name = name
        self.__event = Event()
        self.__event.set() # Set internal flag to true
    
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
    
class Return:
    """
        Saving return values ​​from all functions during Core execution.
        
        Note: that if you use a function multiple times in Core, 
              you must define a different name for each of them in Agent so that their outputs are saved correctly if there is a difference.
            
        
    """
        
    __core_done_counter = {
        "size": 0,
        "counter": 0,
        "workers_added": 0
    } # Saving the number of tasks and completed tasks as well as managing the number of workers in Core.__executor
    
    returns = {} # Store return values
    exists = {} # To quickly check for the presence or absence of a return value
    
    def __init__(self, *args: Any, _set_name: str = "", **kwargs: Any) -> None:
        
        name = _set_name # TODO: Implement a better logic

        if self.exists.get(name, False):
            raise DuplicateKeyError("A duplicate key was found.")
        else:
            self.exists[name] = True
        
        # Store return in self.returns
        if args and kwargs:
            self.returns[name] = {
                "args": args,
                **kwargs
            }
        
        else:
            if args: # Usually only this part is executed.
                self.returns[name] = args[0] if len(args) == 1 else args
            
            if kwargs:
                self.returns[name] = kwargs
      
class Queue:
    """
        Queue with execution time O(1).
        
        This queue uses a dictionary to store values.
        Based on the fact that the keys are a unique and sequential integer and their values ​​are the desired item.
        
        Two counters named __enqueue_counter and __dequeue_counter are used to add or remove values ​​from the queue, 
            check whether the queue is full or empty and control the capacity of the queue if it is defined.
    """
    
    def __init__(self, capacity: int = -1) -> None:
        
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
    
    def __str__(self) -> str:
        return f"Queue({self.__queue})"
           
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

    """
    
    def __init__(self, function: Callable, *args, **kwargs) -> None: 
        
        self.executor: 'ThreadPoolExecutor' = None
        self.name = kwargs.pop("_set_name", function.__name__)
        self.__return_name = kwargs.pop("_return_name", self.name)
        self.args = args
        self.kwargs = kwargs
        self.function = self.__wrapFunction(function)
        self.coroutin = self.__toCoroutine 
    
    def __toCoroutine(self) -> Generator:
        """Submitting a function in the executor and creating a coroutine from it."""
        
        future = self.executor.submit(self.function, *self.args, **self.kwargs)
        return self.__wrapFuture(future=future)
    
    def __wrapFunction(self, function: Callable) -> Callable:
        """Add necessary functionality to the function."""
        
        def wrapper(*args, **kwargs):
            
            result = function(*args, **kwargs)
            
            # Store the return value of the function in the Return class
            Return(result, _set_name=self.__return_name)

            # Management of the number of workers based on the total number of tasks in the Core class
            Return._Return__core_done_counter["counter"] += 1
            
            if Return._Return__core_done_counter["size"] == Return._Return__core_done_counter["counter"]:
                
                # Decrease _max_workers
                Core._Core__executor._max_workers -= Return._Return__core_done_counter["workers_added"]
                                
                # Reset size & counters & workers_added
                Return._Return__core_done_counter["size"] = 0
                Return._Return__core_done_counter["counter"] = 0
                Return._Return__core_done_counter["workers_added"] = 0
        
        return wrapper
    
    def __wrapFuture(self, future: Future) -> Coroutine:
        """Creating a coroutine from a feature"""
        
        def coroutine():
            
            if not future.done():
                yield
            
            return future.result()
        
        return coroutine()
    
    def __str__(self) -> str:
        
        return f"Agent<name: {repr(self.name)}, {self.coroutin.__class__}>"

class Loop:
    """
        Loop is responsible for receiving agents and dropping them on the core.
    """
        
    def __init__(self, *agents: Agent, excecutor: 'ThreadPoolExecutor', list_of_agents: list[Agent] = []) -> None:
        
        self.__agents = list(agents) + list_of_agents
        self.size = len(self.__agents) # Saving the number of tasks (we don't need it now)
        self.queue_agents = Queue(capacity=self.size) # Create a queue with limited capacity. [safe]
        self.excecutor = excecutor

        # Placing the agents in the queue and introducing the executor to them.
        # TODO: This section has been developed in the future to monitor agents in this way.
        for agent in (self.__agents):
            agent.executor = self.excecutor
            self.queue_agents.enqueue(agent)
                     
    def injection(self) -> None:
        """Inject agents into the loop"""
                
        while not self.queue_agents.empty:
            
            agent: Agent = self.queue_agents.dequeue()
            
            if agent is None:
                break

            try:
                
                next(agent.coroutin())
                self.size -= 1
    
            except StopIteration as e: # for safety
                
                break 
                         
class Core:
    """
        The main core that is responsible for receiving, executing and managing functions.
        
    """
    
    # These two variables exist throughout the entire runtime of Core.
    __executor = ThreadPoolExecutor() # The main core that is responsible for receiving, executing and managing functions.
    __controllers: dict[str, Controller] = {} # Saving controllers and using them throughout the program.
        
    def __init__(self, *agents: Agent, list_of_agents: list[Agent] = []) -> None:
        
        self.__agents: list[Agent] = list(agents) + list_of_agents
        self.__agents_size = len(self.__agents)  
        self.loop = Loop(list_of_agents=self.__agents, excecutor=self.__executor) # Set loop
        self.returns = Return.returns # Pointing to the place where the outputs are stored, which of course can be accessed directly from the Return class.
        
        # Increase _max_workers in self.__executor
        self.__workers_size = self.calculate_workers(self.__agents_size)
        self.__executor._max_workers += self.__workers_size
        Return._Return__core_done_counter["size"] += self.__agents_size
        Return._Return__core_done_counter["workers_added"] += self.__workers_size
         
    def run(self):
        """
            Run Core.
            Every time it is executed, it injects existing agents into the loop.
        """
        self.loop.injection()

    def set_controller(self, controller: Controller) -> None:
        """Set the controller with their name in Core.__controllers"""
        
        if self.__controllers.get(controller.name, None):
            raise DuplicateKeyError("This controller has already been added.")
        
        self.__controllers[controller.name] = controller
        
    def __check_controller(self, controller_name: str) -> None:
        """Checking the absence of the controller."""
        
        if controller_name not in self.__controllers:
            raise KeyError("A controller with this name has not been configured.")
    
    def force_stop(self) -> None:
        """An emergency stop of the Core from executing, which causes the entire program to stop."""
        os._exit(0)
    
    def wait_until_pause(self, controller_name: str, timeout: float = None) -> None:
        """Temporarily stops functions that use a shared controller."""
        
        self.__check_controller(controller_name=controller_name)
        self.__controllers[controller_name].wait_until_pause(timeout=timeout)
        
    def pause(self, controller_name: str) -> None:
        """Changing the internal flag of the controller to true to create a pause in the functions."""
        
        self.__check_controller(controller_name=controller_name)
        self.__controllers[controller_name].pause()
        
    def resume(self, controller_name: str) -> None:
        """Changing the internal flag of the controller to false and resuming the execution of functions."""
        
        self.__check_controller(controller_name=controller_name)
        self.__controllers[controller_name].resume()
    
    def is_pause(self, controller_name: str) -> bool:
        """Checking the issuance of the pause command"""
        
        self.__check_controller(controller_name=controller_name)
        return not self.__controllers[controller_name].is_pause()

    def add_task(self, *agents, list_of_agents: list[Agent] = []) -> None:
        """Adding tasks while Core is running and injecting them into the Loop"""
        
        self.__init__(*agents, list_of_agents=list_of_agents)
        self.run()
    
    @classmethod
    def calculate_workers(self, size: int) -> int:
        """
            An unprincipled way to calculate the number of required workers based on the number of tasks and the number of CPU cores.
            
            TODO: Creating a formula based on the number of tasks, their type, CPU conditions and important and basic parameters.
        """

        if size <= 11:
            return 0
        
        return int(log(size / 10, 1.101)) * os.cpu_count()


