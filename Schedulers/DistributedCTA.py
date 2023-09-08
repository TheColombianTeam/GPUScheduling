from utils.args import args


from .models import Scheduler
from .utils import complete, save_csv, save_json


MS = args.mxm.MS
NS = args.mxm.NS
KS = args.mxm.KS


class DistributedCTA(Scheduler.Scheduler):
    def __init__(self):
        super().__init__()
        self._name = "DistributedCTA"
        self.__scheduler_info = []
        self._n_clusters = args.gpu.cluster
        self._n_SM_per_cluster = args.gpu.sm
        self._n_CTA_per_SM = args.gpu.CTAs_buffer
        self.CTAs_to_schedule_statically_in_SM = (
            self._n_SM_per_cluster * self._n_CTA_per_SM
        )

    def scheduler_algorithm(self, a, b, c):
        self.__scheduler_info = []
        self.__tiling(a, b, c)
        to_schedule_CTAs_per_cluster = []
        for c in range(self._n_clusters):
            to_schedule_CTAs_per_cluster.append([])

        total_num_of_CTA = len(self.__scheduler_info)
        left_to_schedule = int(total_num_of_CTA % self._n_clusters)

        already_scheduled_CTA = []
        for c in range(self._n_clusters):
            if c < left_to_schedule:
                additional_CTA = 1
            else:
                additional_CTA = 0
            count = int(total_num_of_CTA / self._n_clusters) + additional_CTA
            for i in range(c):
                count += already_scheduled_CTA[i]
            already_scheduled_CTA.append(count)

        cluster_id = 0
        # CLUSTER SCHEDULING, per-cluster Pool of CTA generation
        for CTA_id in range(total_num_of_CTA):
            if CTA_id < already_scheduled_CTA[cluster_id]:
                self.__scheduler_info[CTA_id]["Cluster"] = cluster_id
                to_schedule_CTAs_per_cluster[cluster_id].append(CTA_id)
            else:
                cluster_id += 1
                self.__scheduler_info[CTA_id]["Cluster"] = cluster_id
                to_schedule_CTAs_per_cluster[cluster_id].append(CTA_id)

        # It goes without saying that this procedure is static per each CTA
        # SM SCHEDULING
        for c in range(self._n_clusters):
            SM_id = 0
            dynamic_scheduled_row = 0
            stocastic_ordering = []

            for CTA in range(len(to_schedule_CTAs_per_cluster[c])):
                if CTA < self.CTAs_to_schedule_statically_in_SM:
                    self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                        "SM"
                    ] = SM_id
                    SM_id += 1
                    if SM_id == self._n_SM_per_cluster:
                        SM_id = 0
                else:  # pseudo-dynamic scheduling
                    if (
                        CTA
                        == self.CTAs_to_schedule_statically_in_SM
                        + dynamic_scheduled_row * self._n_SM_per_cluster
                    ):
                        SM_id = 0
                        stocastic_ordering = []
                        stocastic_ordering = self.random_delay_generator_simulator(
                            self._n_SM_per_cluster
                        )
                    self.__scheduler_info[to_schedule_CTAs_per_cluster[c][CTA]][
                        "SM"
                    ] = stocastic_ordering[SM_id]
                    SM_id += 1
                    if SM_id == self._n_SM_per_cluster:
                        SM_id = 0
                        dynamic_scheduled_row += 1

        save_csv(self.__scheduler_info, "distributed_CTA")
        save_json(self.__scheduler_info, "distributed_CTA")
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
