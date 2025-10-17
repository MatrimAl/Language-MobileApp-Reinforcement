# 🔒 Security Guidelines

## 🚨 NEVER Commit These Files

The following files contain sensitive information and should NEVER be committed to GitHub:

### 🔑 Credentials & Keys
- `.env` - Environment variables with passwords and API keys
- `config.local.py` - Local configuration with secrets
- `secrets.py` / `secrets.json` - Any file with credentials
- `*api_key*`, `*apikey*`, `*secret_key*` - API key files
- `*.pem`, `*.key`, `*.cert` - SSL certificates and private keys

### 📱 Mobile App Secrets
- `google-services.json` - Firebase credentials (Android)
- `GoogleService-Info.plist` - Firebase credentials (iOS)
- `*.jks`, `*.keystore` - Android signing keys
- `*.p8`, `*.p12`, `*.mobileprovision` - iOS certificates

### 💾 Database
- `*.db`, `*.sqlite` - Local database files
- `*.sql`, `*.dump` - Database dumps with user data
- `database_config.py` - Database credentials

### 🤖 ML Models (if large)
- `models/*.pth` - Trained PyTorch models (can be large)
- Training data with personal information

## ✅ How to Protect Your Secrets

### 1. Use `.env` File

Create a `.env` file in `backend/` directory:

```bash
# Copy example file
cp .env.example .env

# Edit with your real values
nano .env
```

**Example:**
```bash
MONGO_URL=mongodb://localhost:27017
SECRET_KEY=super-secret-change-this-in-production
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 2. Load Environment Variables

In Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
API_KEY = os.getenv("OPENAI_API_KEY")
```

### 3. Verify `.gitignore`

Before committing, verify sensitive files are ignored:

```bash
# Check what will be committed
git status

# Test if .env is ignored
git check-ignore .env
# Should output: .env

# List all ignored files
git ls-files --others --ignored --exclude-standard
```

## 🔍 Security Checklist

Before pushing to GitHub:

- [ ] `.env` file is NOT in git (`git status` should not show it)
- [ ] All API keys are in `.env`, not hardcoded
- [ ] `.gitignore` includes all sensitive file patterns
- [ ] `.env.example` exists with dummy values
- [ ] No passwords in commit history (`git log --all -p | grep -i password`)
- [ ] Trained models are excluded (if large > 100MB)
- [ ] No database dumps with real user data

## 🛠️ What to Do If You Accidentally Committed Secrets

### If not pushed yet:
```bash
# Remove from staging
git reset HEAD .env

# Or amend last commit
git commit --amend
```

### If already pushed to GitHub:

1. **Immediately revoke/rotate the compromised credentials**
2. Remove from git history:
```bash
# Remove file from entire history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push origin --force --all
```

3. **Better option:** Use BFG Repo-Cleaner:
```bash
# Install BFG
# Download from: https://rtyley.github.io/bfg-repo-cleaner/

# Remove sensitive file
java -jar bfg.jar --delete-files .env

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push --force
```

4. **GitHub Secret Scanning:** GitHub may detect leaked secrets and send you a warning. Act immediately!

## 🔐 Production Deployment

### Environment Variables

**Never** hardcode secrets in production. Use:

- **Heroku:** Settings → Config Vars
- **AWS:** Systems Manager Parameter Store / Secrets Manager
- **Azure:** Key Vault
- **Docker:** Use secrets or env files (not in image)
- **Vercel/Netlify:** Environment Variables in dashboard

### Example Docker Compose (Production):
```yaml
services:
  backend:
    image: your-app:latest
    environment:
      - MONGO_URL=${MONGO_URL}  # From host .env
      - SECRET_KEY=${SECRET_KEY}
    secrets:
      - db_password

secrets:
  db_password:
    external: true
```

## 📚 Resources

- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

## 📧 Report Security Issues

If you find a security vulnerability, please **DO NOT** open a public issue.

Contact: [Your email or security contact]

---

**Remember:** Once a secret is pushed to GitHub, consider it compromised even if you delete it later. Always rotate credentials immediately.
