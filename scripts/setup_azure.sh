#!/usr/bin/env bash
# Azure setup script — run once to provision the App Service and configure CI/CD.
# Prerequisites: az CLI logged in, GitHub repo exists.

set -euo pipefail

APP_NAME="${AZURE_WEBAPP_NAME:-image-captioning-api}"
RG="image-captioning-rg"
LOCATION="eastus"
PLAN="caption-plan"
GHCR_IMAGE="ghcr.io/${GITHUB_REPOSITORY_OWNER:-<YOUR_GH_USERNAME>}/image-captioning-api:latest"

echo "==> Creating resource group: $RG"
az group create --name "$RG" --location "$LOCATION"

echo "==> Creating App Service Plan (B2 Linux)"
az appservice plan create --name "$PLAN" --resource-group "$RG" \
  --sku B2 --is-linux

echo "==> Creating Web App (container)"
az webapp create --name "$APP_NAME" --resource-group "$RG" \
  --plan "$PLAN" \
  --deployment-container-image-name "$GHCR_IMAGE"

echo "==> Configuring app settings"
az webapp config appsettings set --name "$APP_NAME" --resource-group "$RG" \
  --settings WEBSITES_PORT=5000 DOCKER_ENABLE_CI=true

echo "==> Enabling logging"
az webapp log config --name "$APP_NAME" --resource-group "$RG" \
  --docker-container-logging filesystem

echo ""
echo "Done! Your app will be available at:"
echo "  https://${APP_NAME}.azurewebsites.net"
echo ""
echo "Add these secrets to your GitHub repo (Settings > Secrets):"
echo "  AZURE_CREDENTIALS  — output of: az ad sp create-for-rbac --sdk-auth"
echo "  AZURE_WEBAPP_NAME  — $APP_NAME"
