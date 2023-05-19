import logging

logger = logging.getLogger(__name__)


class TSVReader:
    def __init__(self, file_path, selected_cols=None, separator="\t"):
        fp = open(file_path, encoding='utf-8')
        headers = fp.readline().strip().split(separator)
        if selected_cols is not None:
            col_ids = []
            for v in selected_cols.split(','):
                col_ids.append(headers.index(v))
            selected_cols = col_ids
        else:
            selected_cols = list(range(len(headers)))

        self.contents = []
        for row in fp:
            if selected_cols is not None:
                column_l = row.rstrip("\n").split(separator, len(headers) - 1)
                column_l = [column_l[col_id] for col_id in selected_cols]
            else:
                column_l = row.rstrip("\n").split(separator, len(headers) - 1)
            self.contents.append(column_l)

        logger.info("loaded {}".format(file_path))
        fp.close()

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        column_l = self.contents[index]
        return column_l
