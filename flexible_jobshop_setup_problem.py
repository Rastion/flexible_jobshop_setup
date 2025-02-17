from qubots.base_problem import BaseProblem
import random
import os
# Constant to denote an incompatible machine.
INFINITE = 1000000

class FlexibleJobShopSetupProblem(BaseProblem):
    """
    Flexible Job Shop Scheduling with Sequence‑Dependent Setup Times (FJSP‑SDST) for Qubots.
    
    In this problem, each job consists of an ordered sequence of operations.
    Each operation can be processed on one of several compatible machines, with machine‐dependent processing times.
    Each machine can process only one operation at a time.
    In addition, a setup time is required on a machine between any two consecutive operations,
    and these setup times are sequence dependent.
    
    **Solution Representation:**
      A dictionary mapping each job (0-indexed) to a list of operations.
      Each operation is represented as a dictionary with keys:
        - "machine": the selected machine (0-indexed)
        - "start": the start time of the operation
        - "end": the finish time (which must equal start + processing time on the chosen machine)
    """
    
    def __init__(self, instance_file: str):
        (self.nb_jobs,
         self.nb_machines,
         self.nb_tasks,
         self.task_processing_time_data,
         self.job_operation_task,
         self.nb_operations,
         self.task_setup_time_data,
         self.max_start) = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):
        """
        Reads an instance file with the following format:
        
        - First line: two integers representing the number of jobs and the number of machines 
          (an extra number may be present and is ignored).
          
        - For each job (next nb_jobs lines):
            * The first integer is the number of operations in that job.
            * Then, for each operation:
                  - An integer indicating the number of machines compatible with the operation.
                  - For each compatible machine: a pair of integers (machine id and processing time).
                  (Machine ids are given as 1-indexed.)
        
        - For each machine (next nb_machines × nb_tasks lines):
            * For each machine m, there are nb_tasks lines.
              Each line contains a list of integers representing the setup times on machine m
              from a given preceding task (row) to every other task (columns).
              (A setup time of INFINITE indicates an infeasible sequence.)
        
        The trivial upper bound for the start times is computed as the sum of the maximum processing times
        over all tasks plus nb_tasks times the maximum setup time.
        
        Note: If blank lines are removed (as done here), the index for the first setup line becomes (nb_jobs + 1).
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)


        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Remove blank lines.
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line: number of jobs and machines (ignore extra number if present)
        first_line = lines[0].split()
        nb_jobs = int(first_line[0])
        nb_machines = int(first_line[1])
        
        # Read number of operations for each job.
        nb_operations = [int(lines[j + 1].split()[0]) for j in range(nb_jobs)]
        
        # Total number of tasks (operations)
        nb_tasks = sum(nb_operations[j] for j in range(nb_jobs))
        
        # Processing time for each task on each machine.
        task_processing_time = [[INFINITE for _ in range(nb_machines)] for _ in range(nb_tasks)]
        
        # For each job, store the corresponding task ids for each operation.
        job_operation_task = [[0 for _ in range(nb_operations[j])] for j in range(nb_jobs)]
        
        # Read processing times.
        task_id = 0
        for j in range(nb_jobs):
            parts = lines[j + 1].split()
            tmp = 1  # start after the number of operations
            for o in range(nb_operations[j]):
                nb_machines_op = int(parts[tmp])
                tmp += 1
                for i in range(nb_machines_op):
                    # Machine id is 1-indexed in the file.
                    machine = int(parts[tmp + 2 * i]) - 1
                    time = int(parts[tmp + 2 * i + 1])
                    task_processing_time[task_id][machine] = time
                job_operation_task[j][o] = task_id
                task_id += 1
                tmp += 2 * nb_machines_op
        
        # Read setup times.
        # There are nb_machines blocks, each with nb_tasks lines.
        task_setup_time = [[[ -1 for _ in range(nb_tasks)] for _ in range(nb_tasks)] for _ in range(nb_machines)]
        # Adjusted index: use nb_jobs + 1 (instead of nb_jobs + 2) because blank lines were removed.
        id_line = nb_jobs + 1
        max_setup = 0
        for m in range(nb_machines):
            for i1 in range(nb_tasks):
                setup_values = list(map(int, lines[id_line].split()))
                task_setup_time[m][i1] = setup_values
                max_setup = max(max_setup, max(s if s != INFINITE else 0 for s in setup_values))
                id_line += 1
        
        # Compute a trivial upper bound: sum of maximum processing times plus nb_tasks * max_setup.
        max_sum_processing = sum(
            max(task_processing_time[i][m] for m in range(nb_machines) if task_processing_time[i][m] != INFINITE)
            for i in range(nb_tasks)
        )
        max_start = max_sum_processing + nb_tasks * max_setup
        
        return nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, task_setup_time, max_start
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a dictionary mapping each job (0-indexed) to a list of operations.
          Each operation is a dictionary with keys "machine", "start", and "end".
        
        The evaluation checks:
          - Each operation is scheduled on a compatible machine (processing time ≠ INFINITE).
          - The operation's end time equals its start time plus the processing time.
          - Precedence constraints: within each job, each operation must start after the previous one ends.
          - Disjunctive constraints on each machine: for tasks processed on the same machine,
            the start time of a task must be at least the end time of the preceding task plus the setup time
            from the preceding task to the current task.
        
        Returns:
          - The makespan (maximum completion time over all jobs) if the solution is feasible.
          - A penalty value (1e9) if any constraint is violated.
        """
        penalty = 1e9
        
        # Validate solution structure.
        if not isinstance(solution, dict):
            return penalty
        
        overall_end = 0
        # For machine disjunctive constraints, record intervals per machine as (task_id, start, end).
        machine_intervals = {m: [] for m in range(self.nb_machines)}
        
        # Check job precedence constraints.
        for j in range(self.nb_jobs):
            if j not in solution:
                return penalty
            ops = solution[j]
            if len(ops) != self.nb_operations[j]:
                return penalty
            
            prev_end = None
            for o, op in enumerate(ops):
                if not isinstance(op, dict):
                    return penalty
                for key in ['machine', 'start', 'end']:
                    if key not in op:
                        return penalty
                m = op['machine']
                start = op['start']
                end = op['end']
                if m < 0 or m >= self.nb_machines:
                    return penalty
                task_id = self.job_operation_task[j][o]
                proc_time = self.task_processing_time_data[task_id][m]
                if proc_time == INFINITE:
                    return penalty
                if end != start + proc_time:
                    return penalty
                if prev_end is not None and start < prev_end:
                    return penalty
                prev_end = end
                
                machine_intervals[m].append((task_id, start, end))
                overall_end = max(overall_end, end)
        
        # Check disjunctive machine constraints.
        for m in range(self.nb_machines):
            intervals = sorted(machine_intervals[m], key=lambda x: x[1])
            for i in range(len(intervals) - 1):
                task_id_1, start1, end1 = intervals[i]
                task_id_2, start2, end2 = intervals[i+1]
                setup = self.task_setup_time_data[m][task_id_1][task_id_2]
                if start2 < end1 + setup:
                    return penalty
        
        return overall_end
    
    def random_solution(self):
        """
        Generates a random candidate solution.
        
        For each job, operations are scheduled sequentially.
        For each operation, a random compatible machine is chosen and a start time is assigned
        such that the operation begins no earlier than the previous operation's finish time.
        (Note: This simple generator does not resolve potential machine conflicts across different jobs.)
        """
        solution = {}
        for j in range(self.nb_jobs):
            ops = []
            current_time = random.randint(0, self.max_start // 4)
            for o in range(self.nb_operations[j]):
                task_id = self.job_operation_task[j][o]
                compatible = [m for m in range(self.nb_machines)
                              if self.task_processing_time_data[task_id][m] != INFINITE]
                if not compatible:
                    m = 0
                else:
                    m = random.choice(compatible)
                proc_time = self.task_processing_time_data[task_id][m]
                start = current_time
                end = start + proc_time
                ops.append({
                    "machine": m,
                    "start": start,
                    "end": end
                })
                current_time = end + random.randint(0, self.max_start // 10)
            solution[j] = ops
        return solution
