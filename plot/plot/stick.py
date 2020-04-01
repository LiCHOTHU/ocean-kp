import matplotlib.pyplot as ax

def cutsomStick(game, type_, ax):
    # ax.ylabel('Episode Rewards')
    if type_ == 'timesteps':
        ax.set_xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6])
        ax.set_xticklabels(["1M", "2M", "4M", "6M", "8M", "10M"])
       #  ax.xlabel('Number of Timesteps')

        if game == 'Atlantis':
            ax.xlim(0, 2.6e6)
            ax.xticks([1e6, 2e6, 2.5e6], ["1M", "2M", "2.5M"])
        else:
            ax.set_xlim(0, 10e6)

    if type_ == 'time':
        # ax.xlabel('Hours')

        if game == 'Atlantis':
            ax.set_xlim(0, 1.5)

    if type_ == 'updates':
        # ax.xlabel('Number of Updates')

        if game == 'Atlantis':
            ax.xlim(0, 15000)

        if game == 'Breakout':
            ax.xlim(0, 60000)

    if type_ == 'eposide':
        # ax.xlabel('Number of Episode')

        if game == 'Atlantis':
            ax.xlim(0, 750)

    if game == 'Atlantis':
        ax.yticks([0.5e6, 1e6, 2e6], ['0.5M', '1M', '2M'])

    ax.set_title(game)
    # ax.legend(loc=2)
