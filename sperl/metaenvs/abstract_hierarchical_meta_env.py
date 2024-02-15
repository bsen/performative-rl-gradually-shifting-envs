class HierarchicalMetaEnv:
    """
    An abstract class describing what a hierarchical meta environment should
    implement.
    """

    def __init__(self, setting, random_generator):
        self.setting = setting
        self.random_generator = random_generator

    def change_function(self, policies, num_trajectories, policy_index):
        """
        An implementation of the change functions from the paper.
        The probability transition and reward functions of the underlying
        environment change.

        Args:
            policies -- the policies the environment adopts to
            policy_index -- the index of the policy for which to calculate the
                trajectories.
            num_trajectories -- the number of trajectories (can be set to -1,k
                in which case  the exact probability transition and reward functions
                are returned.

        Returns:
            If the num_trajectories is larger than 0 :
                a list trajectories_over_t of num_trajectories many trajectories
                from the new environment. The list has shape
                num_trajectories x len_trajectory x 3, where
                trajectories_over_t[a,b,c] corresponds to trajectory a of the
                trajectories of this round, the b-th entry in this trajectory and
                if c==0 the state, c==1 the action and c==2 the reward of this entry.
            If the num_trajectories is equal to -1 :
                the current probability transition and reward functions of the
                underlying environment
        """

    def reset(self, random_generator):
        """
        Returns:
            A list of the initial policies
        """
        raise NotImplementedError

    def num_states(self, policy_index):
        pass

    def num_actions(self, policy_index):
        pass

    def start_distribution(self, policy_index):
        pass

    def num_policies(self):
        pass
