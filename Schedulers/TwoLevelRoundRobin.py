from utils.args import args


from .models import Scheduler
from .utils import complete, save_csv, save_json


MS = args.mxm.MS
NS = args.mxm.NS
KS = args.mxm.KS


class TwoLevelRoundRobin(Scheduler.Scheduler):
    def __init__(self):
        super().__init__()
        self._name = "Two-LevelRoundRobin"
        self.__scheduler_info = []
        self._n_clusters = args.gpu.cluster
        self._n_SM_per_cluster = args.gpu.sm
        self._n_CTA_per_SM = args.gpu.CTAs_buffer
        self.CTAs_to_schedule_statically_in_clusters = (
            self._n_SM_per_cluster * self._n_CTA_per_SM * self._n_clusters
        )
        self.CTAs_to_schedule_statically_in_SM = (
            self._n_SM_per_cluster * self._n_CTA_per_SM
        )

    def scheduler_algorithm(self, a, b, c):
        self.__scheduler_info = []
        self.__tiling(a, b, c)

        cluster_id = 0
        to_schedule_CTAs_per_cluster = []
        dynamic_scheduled_row = 0
        stocastic_ordering = []

        for c in range(self._n_clusters):
            to_schedule_CTAs_per_cluster.append([])

        # CLUSTER SCHEDULING
        for CTA_id in range(len(self.__scheduler_info)):
            if CTA_id < self.CTAs_to_schedule_statically_in_clusters:
                to_schedule_CTAs_per_cluster[cluster_id].append(CTA_id)
                self.__scheduler_info[CTA_id]["Cluster"] = cluster_id
                cluster_id += 1
                if cluster_id == self._n_clusters:
                    cluster_id = 0
            else:  # dynamic scheduling
                if (
                    CTA_id
                    == dynamic_scheduled_row * self._n_SM_per_cluster * self._n_clusters
                    + self.CTAs_to_schedule_statically_in_clusters
                ):
                    cluster_id = 0
                    dynamic_scheduled_row += 1
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
                    cluster_id += 1
 
                    if cluster_id == self._n_clusters:
                        cluster_id = 0
                else:
                    to_schedule_CTAs_per_cluster[stocastic_ordering[cluster_id]].append(
                        CTA_id
                    )
                    self.__scheduler_info[CTA_id]["Cluster"] = stocastic_ordering[
                        cluster_id
                    ]
                    cluster_id += 1
                    if cluster_id == self._n_clusters:
                        cluster_id = 0

        # SM SCHEDULING
        for c in range(len(to_schedule_CTAs_per_cluster)):  # for each cluster
            SM_id = 0
            n_CTA_for_this_cluster = len(to_schedule_CTAs_per_cluster[c])
            stocastic_ordering = []
            dynamic_row_scheduled = 0

            for CTA in range(n_CTA_for_this_cluster):
                if CTA < self.CTAs_to_schedule_statically_in_SM:
                    self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                        "SM"
                    ] = SM_id
                    SM_id += 1
                    if SM_id == self._n_SM_per_cluster:
                        SM_id = 0
                else:  # dynamic scheduling
                    if (
                        CTA
                        == self.CTAs_to_schedule_statically_in_SM
                        + dynamic_row_scheduled * self._n_SM_per_cluster
                    ):
                        dynamic_row_scheduled += 1
                        SM_id = 0
                        stocastic_ordering = []
                        stocastic_ordering = self.random_delay_generator_simulator(
                            self._n_SM_per_cluster
                        )
                        self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                            "SM"
                        ] = stocastic_ordering[SM_id]
                        SM_id += 1

                    else:
                        self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                            "SM"
                        ] = stocastic_ordering[SM_id]
                        SM_id += 1
                        if SM_id == self._n_SM_per_cluster:
                            SM_id = 0

        save_csv(self.__scheduler_info, "two_level_round_robin")
        save_json(self.__scheduler_info, "two_level_round_robin")
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
