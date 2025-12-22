from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # print dataset statistics
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        # -------------------------------------------------------------------------
        # 1. Standard Overall Evaluation (Original Code)
        # -------------------------------------------------------------------------
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')

        # -------------------------------------------------------------------------
        # 2. NEW: Performance by Popularity Groups (Quartiles)
        # -------------------------------------------------------------------------
        print("\n" + "="*80)
        print("   PERFORMANCE BY POPULARITY QUARTILES (25% Splits)")
        print("="*80)

        # A. Calculate Item Popularity from Training Set
        item_count = {}
        for user in self.data.training_set_u:
            for item in self.data.training_set_u[user]:
                item_count[item] = item_count.get(item, 0) + 1

        # B. Sort items by popularity (High to Low)
        # We only care about items that actually exist in the training set
        unique_items = list(item_count.keys())
        sorted_items = sorted(unique_items, key=lambda x: item_count[x], reverse=True)
        
        total_items = len(sorted_items)
        split_size = total_items // 4

        # C. Define the 4 Quartiles
        quartiles = {
            "1. Top 25% (Most Popular)":   set(sorted_items[:split_size]),
            "2. 25% - 50%":                set(sorted_items[split_size : split_size*2]),
            "3. 50% - 75%":                set(sorted_items[split_size*2 : split_size*3]),
            "4. Bottom 25% (Least Popular)": set(sorted_items[split_size*3:])
        }

        # D. Loop through each group and evaluate
        for group_name, allowed_items in quartiles.items():
            # Create a Filtered Test Set
            # We keep the user structure, but remove any ground-truth item NOT in the current group
            filtered_test_set = {}
            
            for user in self.data.test_set:
                # Get the user's actual true items
                true_items = self.data.test_set[user]
                # Keep only those that belong to the current popularity group
                filtered_items = [item for item in true_items if item in allowed_items]
                
                # Only add this user to the evaluation if they actually have items in this group
                if len(filtered_items) > 0:
                    filtered_test_set[user] = filtered_items

            if len(filtered_test_set) == 0:
                print(f"\n--- {group_name} ---")
                print("No test items found in this group.")
                continue

            # Evaluate using the existing helper function
            # We pass the SAME recommendations (rec_list), but a filtered Ground Truth (filtered_test_set)
            # The evaluator will simply ignore hits on items that aren't in the filtered truth.
            group_result = ranking_evaluation(filtered_test_set, rec_list, self.topN)
            
            # Print cleanly
            print(f"\n--- {group_name} ---")
            print(f"Users with items in this group: {len(filtered_test_set)}")
            print("".join(group_result).strip())

        print("="*80 + "\n")

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure
