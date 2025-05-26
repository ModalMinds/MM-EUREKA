import heapq
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import OpenAI

OPENAI_API_KEY = 'FAKE_API_KEY'
MODEL_NAME = 'Qwen2.5-72B-Instruct'

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

logger = logging.getLogger('main')


class State:
    def __init__(self, partial_solution, parent=None):
        self.partial_solution = partial_solution
        self.parent = parent
        self.visit_count = 0
        self.mc_value = None
        self.incorrect_rollouts = []
        self.correct_rollouts = []
        self.incorrect_rollouts_backup = []
        self.num_incorrect_rollouts = 0
        self.num_correct_rollouts = 0
        self.children = []

    def add_correct_rollout(self, rollout):
        if rollout not in self.correct_rollouts:
            child = State(partial_solution=rollout, parent=self)
            child.mc_value = 1.0
            self.children.append(child)
            self.correct_rollouts.append(rollout)
        self.num_correct_rollouts += 1

    def add_incorrect_rollout(self, rollout):
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)
            self.incorrect_rollouts_backup.append(rollout)
        self.num_incorrect_rollouts += 1

    def remove_incorrect_rollout(self, rollout):
        if rollout in self.incorrect_rollouts:
            self.incorrect_rollouts.remove(rollout)

    def get_solution_prefix(self):
        if not self.parent:
            return tuple()
        return self.parent.get_solution_prefix() + self.partial_solution

    def get_tree_structure(self):
        return {
            'partial_solution': self.partial_solution,
            'mc_value': self.mc_value,
            'visit_count': self.visit_count,
            'correct_rollouts': self.correct_rollouts,
            'incorrect_rollouts': self.incorrect_rollouts_backup,
            'children': [child.get_tree_structure() for child in self.children],
        }

    def __lt__(self, other):
        return self.visit_count < other.visit_count


class CandidatePool:
    def __init__(self):
        self.heap = []
        self.entries = {}

    def update(self, state, rollout, priority):
        key = (state, rollout)
        self.entries[key] = priority
        heapq.heappush(self.heap, (-priority, state, rollout))
        logger.debug(f'Updating candidate pool {key} with priority: {priority}')

    def pop(self):
        while self.heap:
            negative_priority, state, rollout = heapq.heappop(self.heap)
            entry_priority = self.entries.get((state, rollout))

            if entry_priority is None:
                continue
            if entry_priority == -negative_priority:
                del self.entries[(state, rollout)]
                state.remove_incorrect_rollout(rollout)
                logger.debug(f'Popped rollout: {rollout} from state: {state}')
                return state, rollout
            else:
                continue
        return None, None

    def is_empty(self):
        return not self.entries


class OmegaPRM:
    def __init__(
        self,
        LLM,
        question,
        image_path,
        correct_answer,
        c_puct,
        alpha,
        beta,
        length_scale,
        num_rollouts,
        max_search_count,
        rollout_budget,
        api_endpoint,
    ):
        self.LLM = LLM
        self.question = question
        self.image_path = image_path
        self.correct_answer = correct_answer
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.length_scale = length_scale
        self.num_rollouts = num_rollouts
        self.max_search_count = max_search_count
        self.rollout_budget = rollout_budget

        self.client = OpenAI(
            base_url=api_endpoint,
            api_key=OPENAI_API_KEY,
        )

        self.candidate_pool = CandidatePool()
        self.root = State(partial_solution=(self.question,), parent=None)

        self.search_count = 0
        self.total_rollouts = 0

        self.executor = ThreadPoolExecutor(max_workers=32)

    def run(self):
        self.monte_carlo_estimation(self.root)

        while (
            self.search_count < self.max_search_count
            and self.total_rollouts < self.rollout_budget
            and not self.candidate_pool.is_empty()
        ):
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                break

            self.binary_search_phase(
                selected_state,
                selected_rollout,
                0,
                len(selected_rollout) - 1,
                len(selected_rollout),
            )

            self.maintenance_phase(selected_state)

            self.search_count += 1

        data = self.root.get_tree_structure()
        del data['partial_solution']
        return {
            'question': self.question,
            'answer': self.correct_answer,
            'image_path': self.image_path,
            **data,
        }

    def monte_carlo_estimation(self, state):
        if not state.parent:
            prompt = f'<image>\n Your task is to answer the question below. Give complete reasoning steps before you answer, and when you are ready to answer, use Answer: The final answer is ..\n\nQuestion: {self.question}'
        else:
            solution_prefix = state.get_solution_prefix()
            solution_prefix = list(
                map(lambda x: '<step>' + x.strip() + '</step>', solution_prefix)
            )
            prompt = f"<image>\n Your task is to answer the question below. Give complete reasoning steps before you answer, and when you are ready to answer, use Answer: The final answer is ..\n\nQuestion: {self.question}\nPrevious Steps:{''.join(solution_prefix)}"
        batch_rollouts = self.LLM.generate_results(
            prompt, self.image_path, self.num_rollouts
        )
        batch_rollouts = list(map(tuple, batch_rollouts))

        self.total_rollouts += len(batch_rollouts)
        solutions = list(map(lambda x: x[-1] if x else '', batch_rollouts))

        correctnesses = list(
            self.executor.map(
                lambda solution: self.check_correctness(
                    question=self.question,
                    correct_answer=self.correct_answer,
                    solution=solution,
                ),
                solutions,
            )
        )

        for correctness, rollout in zip(correctnesses, batch_rollouts):
            if correctness:
                state.add_correct_rollout(rollout)
            else:
                state.add_incorrect_rollout(rollout)

        state.mc_value = (
            state.num_correct_rollouts
            / (state.num_incorrect_rollouts + state.num_correct_rollouts)
            if (state.num_incorrect_rollouts + state.num_correct_rollouts) > 0
            else 0.0
        )

        if state.mc_value != 0.0:
            for rollout in state.incorrect_rollouts:
                priority = self.compute_selection_score(state, rollout)
                self.candidate_pool.update(state, rollout, priority)

    def selection_phase(self):
        selected_state, selected_rollout = self.candidate_pool.pop()
        return selected_state, selected_rollout

    def compute_Q(self, state, rollout):
        word_count = len('\n'.join(rollout).split())
        length_penalty = word_count / self.length_scale
        Q_value = (self.alpha ** (1 - state.mc_value)) * (self.beta**length_penalty)
        return Q_value

    def compute_U(self, state):

        def get_visit_count(state):
            count = state.visit_count
            for child in state.children:
                count += get_visit_count(child)
            return count

        total_visit_count = get_visit_count(self.root)
        U_value = self.c_puct * (math.sqrt(total_visit_count)) / (1 + state.visit_count)
        return U_value

    def compute_selection_score(self, state, rollout):
        Q_value = self.compute_Q(state, rollout)
        U_value = self.compute_U(state)
        score = Q_value + U_value
        return score

    def binary_search_phase(self, state, rollout, left, right, length):
        if left > right:
            return

        mid = (left + right) // 2
        partial_solution = rollout[left : mid + 1]

        b_state = State(partial_solution=partial_solution, parent=state)
        state.children.append(b_state)

        if left == right and left == length - 1:
            b_state.mc_value = (
                1.0
                if self.check_correctness(
                    self.question, self.correct_answer, rollout[-1]
                )
                else 0.0
            )
        else:
            self.monte_carlo_estimation(b_state)

            if b_state.mc_value == 0.0:
                self.binary_search_phase(state, rollout, left, mid - 1, length)
            else:
                self.binary_search_phase(b_state, rollout, mid + 1, right, length)

    def maintenance_phase(self, state):
        state.visit_count += 1

        def update_selection_scores(state):
            if state.mc_value != 0.0:
                for rollout in state.incorrect_rollouts:
                    priority = self.compute_selection_score(state, rollout)
                    self.candidate_pool.update(state, rollout, priority)
            for child in state.children:
                update_selection_scores(child)

        update_selection_scores(self.root)

    def check_correctness(self, question, correct_answer, solution):
        prompt = f"""You are given a question, the correct answer and a model's answer. Please determine if the model's answer matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the model's answer and the correct answer.
Only the correctness of the model's answer matters.
Return only "Yes" if the model's answer is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question:
{question}
--------------------------------
Correct Answer:
{correct_answer}
--------------------------------
Model's Answer:
{solution}
--------------------------------"""
        for _ in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': prompt},
                            ],
                        }
                    ],
                    max_tokens=8,
                    temperature=0.5,
                )
                correctness = response.choices[0].message.content

                if 'yes' == correctness.lower().strip():
                    logger.debug(
                        f'Checking correctness: {question}, {correct_answer}, {solution}, {correctness}'
                    )
                    return True
                elif 'no' == correctness.lower().strip():
                    logger.debug(
                        f'Checking correctness: {question}, {correct_answer}, {solution}, {correctness}'
                    )
                    return False
            except Exception as e:
                logger.error(f'Error in check_correctness: {e}')
        logger.debug(
            f'Checking correctness: {question}, {correct_answer}, {solution}, {False}'
        )
        return False
