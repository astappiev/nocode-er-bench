import os
import csv


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label: int = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter=','):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            '''
            lines = []
            for line in reader:
                lines.append(line)
            return lines
            '''
            lines = []
            for no, line in enumerate(reader):
                if no == 0:
                    continue
                lines.append(line)
            return lines


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        # IMPORTANT: the QQP test-set contains no labels, therefore we use the dev-set as test set
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class DeepMatcherProcessor(DataProcessor):
    """Processor for preprocessed DeepMatcher data sets (abt_buy, company, etc.)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")),
            self._read_tsv(os.path.join(data_dir, "tableA.csv")),
            self._read_tsv(os.path.join(data_dir, "tableB.csv")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.csv")),
            self._read_tsv(os.path.join(data_dir, "tableA.csv")),
            self._read_tsv(os.path.join(data_dir, "tableB.csv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")),
            self._read_tsv(os.path.join(data_dir, "tableA.csv")),
            self._read_tsv(os.path.join(data_dir, "tableB.csv")),
            "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, tableA, tableB, set_type):
        """Creates examples for the training and dev sets."""
        tableA = [' '.join(line[1:]) for line in tableA]
        tableB = [' '.join(line[1:]) for line in tableB]

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # guid = "%s-%s" % (set_type, line[0])
            guid = "%s-%s" % (set_type, i)
            try:
                # text_a, text_b, label = line[1:]
                text_a, text_b, label = line

                text_a = tableA[int(text_a)]
                text_b = tableB[int(text_b)]

                '''
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                '''
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples
