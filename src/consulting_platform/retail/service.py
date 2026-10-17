"""RetailService implementation backed by a trained RetailModel."""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    CatalogResponse,
    Mode,
    Product,
    RetailRecommendation,
    RetailRequest,
    RetailResponse,
)
from .model import MIN_CONFIDENCE, MIN_EVIDENCE_COUNT, RetailModel


class UnknownProductError(ValueError):
    """Cart contains SKUs that are not in the catalog; mapped to HTTP 422."""

    def __init__(self, details: list[tuple[str, str]]) -> None:
        super().__init__("unknown product SKU")
        self.details = details


class RetailModelService:
    def __init__(self, model: RetailModel, *, mode: Mode, warnings: list[str] | None = None) -> None:
        self.model = model
        self.mode = mode
        self.warnings = warnings or []

    @classmethod
    def from_artifact(cls, path: Path, *, mode: Mode) -> "RetailModelService":
        return cls(RetailModel.load(path), mode=mode)

    def get_catalog(self, *, request_id: str) -> CatalogResponse:
        return CatalogResponse(
            request_id=request_id,
            mode=self.mode,
            products=[Product(product_id=sku, name=name) for sku, name in self.model.catalog.items()],
        )

    def recommend(self, request: RetailRequest, *, request_id: str) -> RetailResponse:
        cart = list(request.cart_items)
        unknown = [(f"cart_items.{i}", "Unknown product SKU.") for i, sku in enumerate(cart) if sku not in self.model.catalog]
        if unknown:
            raise UnknownProductError(unknown)

        def response(status: str, explanation: str, recommendations: list[RetailRecommendation]) -> RetailResponse:
            return RetailResponse(
                request_id=request_id,
                mode=self.mode,
                status=status,
                cart_items=cart,
                recommendations=recommendations,
                explanation=explanation,
                model_version=self.model.model_version,
                warnings=list(self.warnings),
            )

        if not cart:
            return response("empty_cart", "Add an item to receive evidence-based suggestions.", [])

        n = self.model.n_training
        cart_label = cart[0] if len(cart) == 1 else f"all {len(cart)} cart items"
        suggestions = self.model.recommend(cart, request.top_k)
        if not suggestions:
            return response(
                "no_supported_pattern",
                f"No product was bought together with {cart_label} in at least {MIN_EVIDENCE_COUNT} of the "
                f"{n} training baskets with confidence of at least {MIN_CONFIDENCE:.0%} and lift above 1, "
                "so no suggestion is shown.",
                [],
            )

        recommendations = [
            RetailRecommendation(
                product_id=s.product_id,
                name=self.model.catalog[s.product_id],
                rank=rank,
                score=s.score,
                support=s.support,
                confidence=s.confidence,
                lift=s.lift,
                evidence_count=s.evidence_count,
                antecedent_count=s.antecedent_count,
                reason=(
                    f"{s.evidence_count} of {s.antecedent_count} training baskets containing {cart_label} "
                    f"also contained {s.product_id}; {s.product_id} appears in {s.product_count} of {n} "
                    "training baskets overall."
                ),
            )
            for rank, s in enumerate(suggestions, start=1)
        ]
        return response(
            "ok",
            f"Suggestions come from training baskets that contained {cart_label}.",
            recommendations,
        )
