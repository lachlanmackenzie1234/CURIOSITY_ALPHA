# ALPHA Backup System

This directory contains tools for maintaining reliable backups of the ALPHA project.

## Backup Layers

1. **Git Version Control**
   - Daily development changes
   - Complete history of code changes
   - Local `.git` repository

2. **GitHub Remote Repository**
   - Remote backup of all code
   - Collaboration platform
   - Issue tracking and project management
   - Setup: `git remote add origin <your-github-repo-url>`

3. **Automated Local Backups**
   - Daily compressed backups
   - Excludes temporary files and virtual environments
   - Keeps last 5 backups
   - Location: `~/ALPHA_Backups/`

## Using the Backup System

### Daily Development
```bash
# Regular git commits
git add .
git commit -m "Description of changes"
git push origin main
```

### Creating Manual Backups
```bash
# Run the backup script
./backup_alpha.sh
```

### Automated Backups
To set up daily automated backups:

1. Open crontab:
```bash
crontab -e
```

2. Add this line for daily backups at 2 AM:
```
0 2 * * * /path/to/ALPHA/tools/backup/backup_alpha.sh
```

## Backup Verification

1. **Git Status**
```bash
git status  # Check for uncommitted changes
git log     # View commit history
```

2. **Local Backups**
```bash
ls -l ~/ALPHA_Backups/  # List all backups
```

## Recovery Procedures

1. **Git Recovery**
```bash
git checkout <commit-hash>  # Revert to specific commit
git reset --hard origin/main  # Reset to remote version
```

2. **Local Backup Recovery**
```bash
cd ~/ALPHA_Backups
tar -xzf ALPHA_backup_YYYYMMDD_HHMMSS.tar.gz
```
