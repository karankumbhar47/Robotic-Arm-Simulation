"""Ravens agents package."""

from src.agents.transporter import GoalNaiveTransporterAgent
from src.agents.transporter import GoalTransporterAgent
from src.agents.transporter import NoTransportTransporterAgent
from src.agents.transporter import OriginalTransporterAgent
from src.agents.transporter import PerPixelLossTransporterAgent

names = {
    'transporter': OriginalTransporterAgent,
    'no_transport': NoTransportTransporterAgent,
    'per_pixel_loss': PerPixelLossTransporterAgent,
    'transporter-goal': GoalTransporterAgent,
    'transporter-goal-naive': GoalNaiveTransporterAgent
}
