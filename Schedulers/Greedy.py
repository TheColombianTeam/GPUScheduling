from utils.args import args


from .models import Scheduler
from .utils import complete, save_csv, save_json


MS = args.mxm.MS
NS = args.mxm.NS
KS = args.mxm.KS


class Greedy(Scheduler.Scheduler):
    def __init__(self):
        super().__init__()
        self._name = "Greedy"
        self.__scheduler_info = []
        self._n_clusters = args.gpu.cluster
        self._n_SM_per_cluster = args.gpu.sm
        self._n_CTA_per_SM = args.gpu.CTAs_buffer
        self.n_CTA_in_cluster = self._n_SM_per_cluster * self._n_CTA_per_SM
        self.CTAs_to_schedule_statically_in_clusters = (
            self._n_SM_per_cluster * self._n_CTA_per_SM * self._n_clusters
        )
        self.CTAs_to_schedule_statically_in_SM = (
            self._n_SM_per_cluster * self._n_CTA_per_SM
        )

    def scheduler_algorithm(self, a, b, c):
        self.__tiling(a, b, c)

        to_schedule_CTAs_per_cluster = []
        dynamic_cluster_filled = 0
        stocastic_ordering = []
        cluster_id = 0
        inter_cluster_counter = 0

        for c in range(self._n_clusters):
            to_schedule_CTAs_per_cluster.append([])

        # CLUSTER SCHEDULING
        for CTA_id in range(len(self.__scheduler_info)):
            if CTA_id < self.CTAs_to_schedule_statically_in_clusters:
                if CTA_id < (cluster_id + 1) * self.n_CTA_in_cluster:
                    to_schedule_CTAs_per_cluster[cluster_id].append(CTA_id)
                    self.__scheduler_info[CTA_id]["Cluster"] = cluster_id
                else:
                    cluster_id += 1  # no need to check if cluster_id == self.n_cluster since as it increases to a value generating index error static scheduling is completed
                    to_schedule_CTAs_per_cluster[cluster_id].append(CTA_id)
                    self.__scheduler_info[CTA_id]["Cluster"] = cluster_id
            else:  # dynamic scheduling
                if (
                    CTA_id
                    == self.CTAs_to_schedule_statically_in_clusters
                    + dynamic_cluster_filled
                    * self.CTAs_to_schedule_statically_in_clusters
                ):
                    cluster_id = 0
                    stocastic_ordering = []
                    stocastic_ordering = self.random_delay_generator_simulator(
                        self._n_clusters
                    )
                to_schedule_CTAs_per_cluster[stocastic_ordering[cluster_id]].append(
                    CTA_id
                )
                self.__scheduler_info[CTA_id]["Cluster"] = stocastic_ordering[
                    cluster_id
                ]
                inter_cluster_counter += 1
                if inter_cluster_counter == self.n_CTA_in_cluster:
                    inter_cluster_counter = 0
                    cluster_id += 1
                if cluster_id == self._n_clusters:
                    cluster_id = 0
                    dynamic_cluster_filled += 1

        # SM SCHEDULING
        for c in range(len(to_schedule_CTAs_per_cluster)):
            SM_id = 0
            stocastic_ordering = []
            n_CTA_for_this_cluster = len(to_schedule_CTAs_per_cluster[c])
            dynamic_CTA_in_current_cluster = 0
            dynamic_block = 0

            for CTA in range(n_CTA_for_this_cluster):
                if CTA < self.CTAs_to_schedule_statically_in_SM:
                    self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                        "SM"
                    ] = SM_id
                    SM_id += 1
                    if SM_id == self._n_SM_per_cluster:
                        SM_id = 0
                else:  # dynamic scheduler
                    if (
                        CTA
                        == self.CTAs_to_schedule_statically_in_SM
                        + dynamic_block * self.CTAs_to_schedule_statically_in_SM
                    ):
                        SM_id = 0
                        stocastic_order = []
                        stocastic_order = self.random_delay_generator_simulator(
                            self._n_SM_per_cluster
                        )

                    self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                        "SM"
                    ] = stocastic_order[SM_id]
                    SM_id += 1
                    dynamic_CTA_in_current_cluster += 1
                    if SM_id == self._n_SM_per_cluster:
                        SM_id = 0
                    if dynamic_CTA_in_current_cluster == self.n_CTA_in_cluster:
                        dynamic_CTA_in_current_cluster = 0
                        dynamic_block += 1

        save_csv(self.__scheduler_info, "greddy")
        save_json(self.__scheduler_info, "greddy")
        return self.__scheduler_info

    def __tiling(self, a, b, c):
        # Initial values
        CTA_id = 0
        SM_id = -1
        cluster_id = -1
        a = complete(a, MS, KS)
        b = complete(b, KS, NS)
        c = complete(c, MS, NS)
        c_shape = c.shape
        for row_c in range(c_shape[0] // MS):
            new_row_c_start = MS * row_c
            for column_c in range(c_shape[1] // NS):
                new_column_c_start = NS * column_c
                CTA_info = {
                    "Cluster": cluster_id,
                    "SM": SM_id,
                    "CTA": {
                        "id": CTA_id,
                        "x": new_row_c_start,
                        "y": new_column_c_start,
                    },
                }
                CTA_id += 1
                self.__create_dict(CTA_info)

    def __create_dict(self, CTA_info):
        self.__scheduler_info.append(CTA_info)

    def read_name(self):
        return self._name
