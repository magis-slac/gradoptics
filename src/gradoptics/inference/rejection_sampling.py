import torch
import gradoptics as optics


def rejection_sampling(pdf, nb_point, proposal_distribution, m=None, batch_size=int(1e6), eps=1e-15, device='cpu'):
    """
    Samples from an unnormalized pdf using rejection sampling

    :param pdf: Function that evaluates the unnormalized pdf of a data point (:py:class:`~Callable`)
    :param nb_point: Number of points to sample (:obj:`int`)
    :param proposal_distribution: Proposal distribution (:py:class:`~gradoptics.optics.distributions.BaseDistribution`)
    :param m: M (:obj:`float`)
    :param batch_size: Batch size (:obj:`int`)
    :param eps: Parameter used for numerical stability (:obj:`float`). Default
                is ``'1e-15'``
    :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

    :return: Sampled points (:obj:`torch.tensor`)
    """
    accepted_data = torch.tensor([], device=device)

    while accepted_data.shape[0] < nb_point:

        # 1: sample U from [0, 1]
        u = torch.rand(batch_size, device=device)

        # 2: sample data from the proposal
        proposed_data = proposal_distribution.sample(batch_size, device=device)

        # If M is not specified, set it empirically
        if m is None:  # @Todo, share M over the different loops! (in order to keep the global extremum)
            m = (pdf(proposed_data) / proposal_distribution.pdf(proposed_data)).max()

        # 3: accept data x if u < pdf(x)/(M * pdf_proposal(x))
        condition = u <= (pdf(proposed_data) / (m * proposal_distribution.pdf(proposed_data) + eps))
        tmp_accepted_data = proposed_data[condition]
        accepted_data = torch.cat((accepted_data, tmp_accepted_data))
        # acceptance_rate = tmp_accepted_data.shape[0] / batch_size * 100

        del u
        del proposed_data
        del condition
        del tmp_accepted_data
        torch.cuda.empty_cache()
    return accepted_data[:nb_point]
