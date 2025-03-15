from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import deque
from threading import Event
from waiting import wait
from math import log
import threading
import traceback
import ctypes
import time
import os


# Contact: mmji-programming@proton.me
# Core execution for concurrently blocking functions
# TODO: Add documentation
# TODO: Add tests
# TODO: Add more examples
# TODO: Add more features
# TODO: Add more optimizations
# TODO: Add compatibility with projects that have a main loop


# Special Errors
class DuplicateKeyError(Exception):
    """Found a key that already exists"""
    __slots__ = () # Optimize memory usage
    __module__ = Exception.__module__

class NoSetController(Exception):
    """No controller is configured with this name."""
    __slots__ = ()
    __module__ = Exception.__module__



# Main classes
class Time:
    __slots__ = ('now',)  # Optimize memory usage
    
    def __init__(self) -> None:
        """Capture current monotonic time"""
        self.now: float = time.monotonic()
    
    def passed(self) -> float:
        """Elapsed time"""
        return time.monotonic() - self.now

    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        
        return f"{self.__get_name()}(Start at: {self.now} [monotonic])"
    
class Queue:
    """
    Optimized Queue implementation using collections.deque for O(1) operations
    """
    __slots__ = ('size', '_queue', '_capacity', 'empty')  # Optimize memory usage
    
    def __init__(self, capacity: int = -1) -> None:
        self.size: int = 0
        self._queue: deque = deque()
        self._capacity: int = capacity
        self.empty: bool = True

    def enqueue(self, item: Any) -> None:
        """Add an item to the queue."""
        if self._capacity > -1 and self.size >= self._capacity:
            return
        
        self._queue.append(item)
        self.size += 1
        self.empty = False
    
    def dequeue(self) -> Any:
        """Remove an item from the queue."""
        if not self._queue:
            self.empty = True
            return None

        self.size -= 1
        if self.size == 0:
            self.empty = True
        return self._queue.popleft()
    
    def get(self, index: int) -> Optional[Any]:
        try:
            return self._queue[index]
        except IndexError:
            return None
    
    def __get_name(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return f"{self.__get_name()}({list(self._queue)}, {'inf' if self._capacity == -1 else self._capacity})"
 
class Label:
    """Labels for each event"""
    
    __slots__ = ('object', 'label', 'args', 'kwargs')  # Optimize memory usage
    
    def __init__(self, _object: Any, label: str, *args, **kwargs) -> None:
        self.object = _object
        self.label = label
        self.args = args
        self.kwargs = kwargs
    
    def __str__(self) -> str:
        try:
            args_str = f", {', '.join(str(i) for i in self.args)}" if self.args else ""
            kwargs_str = f", {', '.join(f'{key}: {repr(value)}' for key, value in self.kwargs.items())}" if self.kwargs else ""
            return f"{self.__class__.__name__}(Object: {repr(str(self.object))}, Label: {repr(self.label)}{args_str}{kwargs_str})"
        except:
            return f"{self.__class__.__name__}(Object: {self.object}, Label: {repr(self.label)})"

class ExecutionTrack:
    """Storing events sequentially with optimized memory usage"""
    
    __slots__ = ()  # No instance attributes needed
    events: Queue = Queue(capacity=-1)
    number_of_core: int = 0
    
    @classmethod
    def add_event(cls, event: Label) -> None:
        """Add event to tracking queue"""
        cls.events.enqueue(event)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(NumberOfCore: {self.number_of_core}, Events: {self.events})"

class Controller:
    """
    Create a 'threading.Event' for one or a set of functions to control them when the main loop of the program is executed.
    A name must be set for the controller.
    """
    
    __slots__ = ('_terminate', 'name', '_event')  # Optimize memory usage
    
    def __init__(self, name: str) -> None:
        self._terminate: bool = False
        self.name: str = name
        self._event: Event = Event()
        self._event.set()  # Set internal flag to true
    
    def terminate(self) -> None:
        """Terminate the execution of functions that use this controller"""
        self._terminate = True
    
    def is_terminate(self) -> bool:
        return self._terminate
    
    def wait_until_pause(self, timeout: Optional[float] = None) -> None:
        """
        Pause execution until resumed or timeout.
        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.
        """
        self._event.wait(timeout=timeout)
    
    def pause(self) -> None:
        """Pause execution of one or a set of functions by this controller."""
        self._event.clear()
    
    def resume(self) -> None:
        """Resume execution of one or set of functions by this controller."""
        self._event.set()
    
    def is_pause(self) -> bool:
        """Returns True if the controller is paused, otherwise returns False."""
        return not self._event.is_set()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(Name: {self.name})"

class Return:
    
    """
    Optimized class for storing return values from all functions during Core execution.
    """
    
    __slots__ = ('core_name', 'args', 'kwargs')
    
    # Class variables with type hints
    _core_done_counter: Dict[str, int] = {
        "size": 0,
        "counter": 0,
        "workers_added": 0,
        "done_counter": 0
    }
    
    returns: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, core_name: str, *args: Any, _set_name: str = "", **kwargs: Any) -> None:
        self.core_name = core_name
        self.args = args
        self.kwargs = kwargs
        
        if not self.returns.get(core_name, {}).get(_set_name, False):
            self.returns[core_name] = self.returns.get(core_name, {})
            self.returns[core_name][_set_name] = {"exec_numb": 0}
        
        self.returns[core_name][_set_name]["exec_numb"] += 1
        exec_numb = self.returns[core_name][_set_name]["exec_numb"]
        
        if kwargs:
            
            self.returns[core_name][_set_name][str(exec_numb)] = {
                "kwargs": kwargs,
                "args": args[0] if len(args) == 1 else args
            }
            
        else:
            self.returns[core_name][_set_name][str(exec_numb)] = args[0] if len(args) == 1 else args
    
    @classmethod
    def reset_counters(cls) -> None:
        """Reset all counters to initial state"""
        cls._core_done_counter.update({
            "size": 0,
            "counter": 0,
            "workers_added": 0,
            "done_counter": 0
        })
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(Parameters({self._core_done_counter}), ReturnsLength({len(self.returns)}))"

class Agent:
    
    """
    Optimized Agent class for managing function execution in the Core.
    """
    
    __slots__ = ('name', '_return_name', '_use_cache', '_cache_size', '_typed', 'function', 'args', 'kwargs', '_core_name')
    
    def __init__(self, function: Callable, *args: Any, **kwargs: Any) -> None:
        # Extract special kwargs
        self.name: str = kwargs.pop("_set_name", function.__name__)
        self._return_name: str = kwargs.pop("_return_name", self.name)
        self._use_cache: bool = kwargs.pop("_use_cache", False)
        self._cache_size: int = kwargs.pop("_cache_size", 128)
        self._typed: bool = kwargs.pop("_cache_typed", False)
        self._core_name: Optional[str] = None
        
        # Setup function
        if self._use_cache:
            
            self.function = self._wrap_function(
                self._use_cache_(function, maxsize=self._cache_size, typed=self._typed)
            )
            
        else:
            self.function = self._wrap_function(function)
        
        self.args = args
        self.kwargs = kwargs
    
    @staticmethod
    def _use_cache_(function: Callable, maxsize: Optional[int], typed: bool) -> Callable:
        """Apply LRU cache to function"""
        return lru_cache(maxsize=maxsize, typed=typed)(function)
    
    def _wrap_function(self, function: Callable) -> Callable:
        """Add necessary functionality to the function."""
        
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            
            if not self._core_name:
                raise RuntimeError("Agent not properly initialized with core name")
                
            ExecutionTrack.add_event(Label(self, "START_AGENT", core_name=self._core_name))
            start = Time()
            
            try:
                result = function(*args, **kwargs)
                
                if self._use_cache:
                    function.cache_clear()
                
                Core._executors[self._core_name]["size"] -= 1
                Return._core_done_counter["counter"] += 1
                
                exec_time = start.passed()
                ExecutionTrack.add_event(Label(self, "END_AGENT", exec_time=exec_time, _return=result, core_name=self._core_name))
                
                # Store return value
                Return(self._core_name, result, _set_name=self._return_name, exec_time=exec_time)
                
                # Manage workers
                if Return._core_done_counter["size"] == Return._core_done_counter["counter"]:
                    
                    Return._core_done_counter["done_counter"] += 1
                    ExecutionTrack.add_event(Label(self._core_name, "END_CORE", last_agent=self.name))
                    
                    # Decrease workers
                    Core._executors[self._core_name]["executor"]._max_workers -= Return._core_done_counter["workers_added"]
                    
                    # Reset counters
                    Return._core_done_counter.update({
                        "size": 0,
                        "counter": 0,
                        "workers_added": 0
                    })
                    
                    Core._executors[self._core_name]["size"] = 0
                    ExecutionTrack.add_event(Label("Return._core_done_counter", "UPDATE_COUNTERS"))
                    
                    # Update execution time
                    Return.returns[self._core_name]["exec_time"] += Core._cores_time[self._core_name]["time"].passed()
                    Core._cores_time[self._core_name]["time"] = Time()
                
                return result
                
            except Exception as e:
                
                exec_time = start.passed()
                Return(self._core_name, str(e), _set_name=self._return_name, exec_time=exec_time)
                ExecutionTrack.add_event(Label(self, "EXCEPTION_AGENT", error=e, exec_time=exec_time, _return=None, core_name=self._core_name))
                
                print(traceback.format_exc())
                ExecutionTrack.add_event(Label(self._core_name, "TERMINATE", error=e, by=f"{self.__class__.__name__}({repr(self.name)})"))
                
                if self._use_cache:
                    function.cache_clear()
                
                Core._terminators[self._core_name]()
                raise
        
        return wrapper
    
    def set_core_name(self, name: str) -> None:
        """Set the core name for this agent"""
        self._core_name = name
    
    def __str__(self) -> str:
        parts = [
            f"Name: {repr(self.name)}",
            f"_return_name: {repr(self._return_name)}"
        ]
        
        if self._use_cache:
            parts.extend([
                f"_use_cache: {self._use_cache}",
                f"_cache_size: {self._cache_size}",
                f"_cache_typed: {self._typed}"
            ])
        
        if self._core_name:
            parts.append(f"CoreName: {repr(self._core_name)}")
        
        parts.extend([
            f"FuncArgs: {self.args}",
            f"FuncKwargs: {self.kwargs}"
        ])
        
        return f"{self.__class__.__name__}({', '.join(parts)})"

class Loop:
    
    """
    Optimized Loop class for managing agent execution in the core.
    """
    
    __slots__ = ('_agents', 'size', 'executor', 'core_name')
    
    def __init__(self, *agents: Agent, executor: ThreadPoolExecutor, core_name: str, list_of_agents: List[Agent] = None) -> None:
        
        self._agents: List[Agent] = list(agents) + (list_of_agents or [])
        self.size: int = len(self._agents)
        self.executor: ThreadPoolExecutor = executor
        self.core_name: str = core_name

    def injection(self, name: str) -> None:
        """Inject agents into the executor for processing"""
        ExecutionTrack.add_event(Label(name, "START_INJECTION"))

        while self.size:
            agent = self._agents.pop(0)  # Get an Agent
            agent.set_core_name(name)  # Set core name for agent
            
            ExecutionTrack.add_event(Label(agent, "INJECT", core_name=name))
            
            self.executor.submit(agent.function, *agent.args, **agent.kwargs)
            self.size -= 1

        ExecutionTrack.add_event(Label(name, "END_INJECTION"))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(Core: {self.core_name})"

class Core:
    """
    Optimized main core class responsible for managing function execution.
    """
    
    # Class variables with type hints
    _controllers: Dict[str, Controller] = {}
    _executed_once_core: Dict[int, bool] = {}
    _executors: Dict[str, Dict[str, Any]] = {}
    _terminates: Dict[str, bool] = {}
    _cores_time: Dict[str, Dict[str, Time]] = {}
    _terminators: Dict[str, Callable] = {}
    _main_thread: Optional[int] = threading.main_thread().ident
    
    __slots__ = ('_executor', '_agents', '_agents_size', '_core_name', '_workers_size', 'loop', 'returns')
    
    def __init__(self, *agents: Agent, list_of_agents: List[Agent] = None, name: str = "") -> None:
        self._executor = ThreadPoolExecutor()
        self._agents: List[Agent] = list(agents) + (list_of_agents or [])
        self._agents_size = len(self._agents)
        
        # Initialize core with unique name
        _id = id(self)
        if _id not in self._executed_once_core:
            ExecutionTrack.add_event(Label(self, "INIT_CORE"))
            
            self._executed_once_core[_id] = True
            self._core_name = name or f"__core_{ExecutionTrack.number_of_core}"
            ExecutionTrack.number_of_core += 1
            
            self._terminates[self._core_name] = False
            self._executors[self._core_name] = {
                "executor": self._executor,
                "size": 0,
                "workers_added": 0
            }
            self._cores_time[self._core_name] = {"time": Time()}
            self._terminators[self._core_name] = self.force_terminate
            Return.returns[self._core_name] = {"exec_numb": 0, "exec_time": 0}
        
        # Configure workers
        self._workers_size = self.calculate_workers(self._agents_size)
        self._executors[self._core_name]["executor"]._max_workers += self._workers_size
        
        # Update counters
        Return._core_done_counter["size"] += self._agents_size
        self._executors[self._core_name]["size"] += self._agents_size
        Return._core_done_counter["workers_added"] += self._workers_size
        self._executors[self._core_name]["workers_added"] += self._workers_size
        
        # Initialize loop
        self.loop = Loop(
            list_of_agents=self._agents,
            executor=self._executors[self._core_name]["executor"],
            core_name=self._core_name
        )
        self.returns = Return.returns
    
    def run(self) -> None:
        """Run core and inject agents into the loop."""
        
        if not self._terminates[self._core_name]:
            Return.returns[self._core_name]["exec_numb"] += 1
            ExecutionTrack.add_event(Label(self, "START_CORE"))
            self.loop.injection(self._core_name)
    
    def set_controller(self, controller: Controller) -> None:
        
        """Set controller with name in Core._controllers"""
        if controller.name in self._controllers:
            ExecutionTrack.add_event(Label(self, "EXCEPTION", exception="This controller has already been added."))
            raise DuplicateKeyError("This controller has already been added.")
        
        ExecutionTrack.add_event(Label(self, "SET_CONTROLLER", repr(str(controller))))
        self._controllers[controller.name] = controller
    
    def _check_controller(self, controller_name: str) -> None:
        """Check if controller exists"""
        if controller_name not in self._controllers:
            ExecutionTrack.add_event(Label(self, "EXCEPTION", exception="A controller with this name has not been configured."))
            raise KeyError("A controller with this name has not been configured.")
    
    def terminate(self, controller_name: str) -> None:
        """Terminate running controller"""
        self._check_controller(controller_name)
        ExecutionTrack.add_event(Label(controller_name, "TERMINATE_CONTROLLER", core_name=self._core_name))
        self._controllers[controller_name].terminate()
    
    def is_terminate(self, controller_name: str) -> bool:
        """Check if controller is terminated"""
        self._check_controller(controller_name)
        return self._controllers[controller_name].is_terminate()
    
    def wait_until_pause(self, controller_name: str, timeout: Optional[float] = None) -> None:
        """Wait until controller is paused or timeout occurs"""
        self._check_controller(controller_name)
        ExecutionTrack.add_event(Label(controller_name, "WAIT_CONTROLLER", core_name=self._core_name))
        self._controllers[controller_name].wait_until_pause(timeout=timeout)
    
    def pause(self, controller_name: str) -> None:
        """Pause controller execution"""
        self._check_controller(controller_name)
        ExecutionTrack.add_event(Label(controller_name, "PAUSE_CONTROLLER", core_name=self._core_name))
        self._controllers[controller_name].pause()
    
    def resume(self, controller_name: str) -> None:
        """Resume controller execution"""
        ExecutionTrack.add_event(Label(controller_name, "RESUME_CONTROLLER", core_name=self._core_name))
        self._controllers[controller_name].resume()
    
    def is_pause(self, controller_name: str) -> bool:
        """Check if controller is paused"""
        self._check_controller(controller_name)
        return not self._controllers[controller_name].is_pause()
    
    def add_task(self, *agents: Agent, list_of_agents: List[Agent] = None) -> None:
        """Add tasks while Core is running"""
        if not self._terminates[self._core_name]:
            ExecutionTrack.add_event(Label(self, "ADD_AGENT_RUNNING", *(list(agents) + (list_of_agents or []))))
            self.__init__(*agents, list_of_agents=list_of_agents)
            self.run()
    
    def force_terminate(self) -> None:
        """Force terminate core execution"""
        if not self._terminates.get(self._core_name, False):
            Return._core_done_counter["done_counter"] += 1
            self._terminates[self._core_name] = True
            
            executor_object = self._executors[self._core_name]
            executor_object["executor"].shutdown(wait=False, cancel_futures=True)
            threads = executor_object["executor"]._threads
            
            # Update counters
            size = executor_object["size"]
            workers_added = executor_object["workers_added"]
            
            # Reset executor state
            self._executors[self._core_name].update({
                "size": 0,
                "workers_added": 0,
                "executor": None
            })
            
            # Update global counters
            Return._core_done_counter["size"] -= size
            Return._core_done_counter["workers_added"] -= workers_added
            Return.returns[self._core_name]["exec_time"] += self._cores_time[self._core_name]["time"].passed()
            self._cores_time[self._core_name]["time"] = Time()
            
            ExecutionTrack.add_event(Label(self, "UPDATE_COUNTERS"))
            
            # Terminate threads
            for thread in threads:
                thread_id = thread.ident
                
                if thread_id != self._main_thread: # Ensuring that the main loop is not stopped
                    
                    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id),
                        ctypes.py_object(SystemExit)
                    )
                    
                    if result > 1:
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
    
    @classmethod
    def force_terminate_Core(cls) -> None:
        """Force terminate all cores"""
        
        ExecutionTrack.add_event(Label(str(cls), "FORCE_TERMINATE"))
        
        for core_name in cls._terminators:
            cls._terminators[core_name]()
            ExecutionTrack.add_event(Label(core_name, "TERMINATE"))
    
    @classmethod
    def is_end(cls) -> bool:
        """Check if all cores have finished execution"""
        
        if Return._core_done_counter["done_counter"] == ExecutionTrack.number_of_core:
            Return.reset_counters()
            return True
        
        return False
    
    @classmethod
    def wait_until_end(cls, period: float = 0.9) -> None:
        """Wait until all cores finish execution"""
        wait(cls.is_end, sleep_seconds=period)
    
    @staticmethod
    def calculate_workers(size: int) -> int:
        """Calculate optimal number of workers based on task size and CPU cores"""
        if size <= 11:
            return 0
        return int(log(size / 10, 1.101)) * os.cpu_count()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(Name: {repr(self._core_name)})"
