import numpy as np


class DenseInputCombineLayer:
    def __init__(self, field_sizes):
        self._field_sizes = field_sizes

    @property
    def output_dim(self):
        return sum(in_dim for _,in_dim in self._field_sizes)

    def forward(self,inputs):

        outputs=[]
        for field_name,in_dim in self._field_sizes:
            a_input=np.asarray(inputs[field_name])
            assert in_dim==a_input.shape[1]
            outputs.append(a_input)
        return np.hstack(outputs)

class SparseInput:
    def __init__(self,n_total_examples,example_indices,feature_ids,feature_values):
        assert len(example_indices)==len(feature_ids)==len(feature_values)
        self._example_indices=example_indices
        self._feature_ids=feature_ids
        self._feature_values=feature_values

        self.n_total_examples=n_total_examples

        self.__nnz_idx=0

    def add(self,example_idx,feat_id,feat_val):
        self._example_indices.append(example_idx)
        self._feature_ids.append(feat_id)
        self._feature_values.append(feat_val)

    def iterate_non_zeros(self):
        return zip(self._example_indices,self._feature_ids,self._feature_values)

    def __move_to_next_example(self,nnz_idx):
        if nnz_idx>=len(self._example_indices):
            return None

        end=nnz_idx+1

        while end<len(self._example_indices) and self._example_indices[end]==self._example_indices[nnz_idx]:
            end+=1
        current_feat_ids=self._feature_ids[nnz_idx:end]
        current_feat_vals=self._feature_values[nnz_idx:end]

        return end,current_feat_ids,current_feat_vals

    def get_example_in_order(self,example_idx):
        if self.__nnz_idx>=len(self._example_indices):
            return [],[]

        elif self._example_indices[self.__nnz_idx]==example_idx:
            self.__nnz_idx,feat_ids,feat_vals=self.__move_to_next_example(self.__nnz_idx)
            return feat_ids,feat_vals

        elif self._example_indices[self.__nnz_idx]>example_idx:
            return [],[]

        else:
            raise ValueError("incorrect invocation")


def get_example_in_order_from_sparse(example_indices, batch_size):
    sp_input = SparseInput(example_indices=example_indices,
                           feature_ids=example_indices,
                           feature_values=example_indices,
                           n_total_examples=batch_size)

    for example_idx in range(batch_size):
        feat_ids, feat_vals = sp_input.get_example_in_order(example_idx)
        print("\n**************** {}-th example: ".format(example_idx))
        print("feature ids:    {}".format(feat_ids))
        print("feature values: {}".format(feat_vals))

get_example_in_order_from_sparse(example_indices=[1, 1, 1, 3, 4, 6],batch_size=10)