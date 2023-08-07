import ast
import re
import rpyc

from multiprocessing import Process
from subprocess import Popen, PIPE

from typing import Dict, Tuple

from intercode.envs.ic_env import (
    IntercodeEnv,
    ACTION_EXEC, AGENT_OBS, EVAL_OBS
)

HOST_PORT = 3006
RESET_KEYWORD = "RESET_CONTAINER_SPECIAL_KEYWORD"

class PythonEnv(IntercodeEnv):
    """Gym environment for python shell"""
    name = "ic_python"

    def __init__(self, image_name: str, **kwargs):
        kwargs['ports'] = {f"{HOST_PORT}/tcp": HOST_PORT}
        super(PythonEnv, self).__init__(image_name, **kwargs)
        self.conn = rpyc.connect("localhost", HOST_PORT)
    
    def reset_container(self) -> None:
        self.conn.root.execute(RESET_KEYWORD)
    
    def exec_action(self, action: str) -> None:
        try:
            if action.strip().startswith("def "):
                function_definition = self.input_multiline_function()
                action = action + "\n" + function_definition
            else:
                action = self.wrap_with_print(action)
            self.observation = self.conn.root.execute(action)
            self.info[ACTION_EXEC] = 'error' in self.observation and len(self.observation['error']) > 0
        except Exception as err:
            self.observation = f"Error executing action: {err}"
            self.info[ACTION_EXEC] = False
    
    def get_reward(self) -> Tuple[float, Dict]:
        self.info = {}

        # Get function from `submit` action
        # TODO: Assert that function name is given upon `submit` action
        last_action = self.trajectory[-1][0]
        func_name = last_action.split(" ")[1]

        # Get gold function name, assign to submitted function
        func_name_ref = re.match(r'def (\w+)\(', self.gold).group(1)
        self.conn.root.execute(f"{func_name_ref} = {func_name}")

        # Run tests against submitted function
        results_pred = []
        self.conn.root.execute(self.record["extra"]["test_setup_code"])
        for test in self.record["extra"]["tests"]:
            results_pred.append({
                "test": test,
                "result": self.conn.root.execute(test)
            })

        # Load gold + run tests
        results_gold = []
        self.conn.root.execute(RESET_KEYWORD)
        self.conn.root.execute(self.record["extra"]["test_setup_code"])
        self.conn.root.execute(self.gold)
        for test in self.record["extra"]["tests"]:
            results_gold.append({
                "test": test,
                "result": self.conn.root.execute(test)
            })
        
        self.info["submitted_function"] = func_name
        self.info[AGENT_OBS] = results_pred
        self.info[EVAL_OBS] = results_gold
        return 0.0, self.info
    
    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent container stopped")
    
    ############################
    ### MARK: Helper methods ###
    ############################
    def input_multiline_function(self):
        lines = []
        while True:
            line = input(". ")
            if len(line) == 0:
                break
            lines.append(line)
        return "\n".join(lines)
    
    def wrap_with_print(self, command):
        # Parse the command as an AST (Abstract Syntax Tree)
        parsed_command = ast.parse(command.strip())

        # Check if the command contains an assignment node, print node, or import
        has_assignment = any(isinstance(node, ast.Assign) for node in ast.walk(parsed_command))
        has_print = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print' for node in ast.walk(parsed_command))
        has_import = any(isinstance(node, ast.Import) for node in ast.walk(parsed_command))

        # Wrap the command with "print" if it's not an assignment and does not have a "print" statement
        if not any([has_assignment, has_print, has_import]):
            return f"print({command})"
        else:
            return command